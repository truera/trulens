"""LLM Jury — ensemble multiple LLM judges into a single feedback callable.

A single LLM judge is noisy and subject to intra-model bias. Ensembling a
*panel* of diverse judges (a "jury") improves reliability, reduces bias, and
can be cheaper when smaller models are used.

"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import inspect
import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

_BUILTIN_STRATEGIES = frozenset({
    "mean",
    "median",
    "trimmed_mean",
    "majority_vote",
    "weighted_mean",
})


class Jury:
    """Ensemble multiple LLM judges into a single feedback callable.

    ``Jury`` wraps N provider instances, calls the same named method on each
    in parallel, and aggregates their scores using a configurable strategy.
    Because ``Jury.__call__`` exposes the same parameter names as the
    underlying provider method, it plugs directly into
    ``Metric(implementation=jury)`` — no changes to Metric, Selector, or
    the evaluation pipeline are needed.

    ``__call__`` always returns ``(score, {"reason": ...})``, matching the
    ``_with_cot_reasons`` convention so per-juror breakdowns flow into
    ``FeedbackCall.meta["reason"]`` and are visible in OTEL spans and the
    dashboard without any UI changes.

    Args:
        jurors: Non-empty list of ``LLMProvider`` instances.
        method: Name of the feedback method to call on each juror, e.g.
            ``"relevance"`` or ``"groundedness_measure_with_cot_reasons"``.
        aggregation: How to combine individual juror scores. Accepts a
            strategy name (``"mean"``, ``"median"``, ``"trimmed_mean"``,
            ``"majority_vote"``, ``"weighted_mean"``) or any
            ``Callable[[list[float]], float]``. Defaults to ``"mean"``.
        weights: Per-juror weights for ``"weighted_mean"``. Must have the
            same length as *jurors*. When a juror fails its weight is
            redistributed proportionally among the successful ones.
        threshold: Binarisation threshold for ``"majority_vote"``.
            Scores >= *threshold* count as a positive vote. Defaults to
            ``0.5``. On an exact tie falls back to median.
        max_workers: Maximum parallel threads. Defaults to
            ``len(jurors)``.

    Example::

        from trulens.core import Metric
        from trulens.feedback.jury import Jury
        from trulens.providers.openai import OpenAI
        from trulens.providers.litellm import LiteLLM

        jury = Jury(
            jurors=[
                OpenAI(model_engine="gpt-4o-mini"),
                OpenAI(model_engine="gpt-4.1-mini"),
                LiteLLM(model_engine="anthropic/claude-3-haiku-20240307"),
            ],
            method="relevance",
            aggregation="median",
        )
        m = Metric(implementation=jury, name="Jury Relevance").on_input().on_output()
    """

    def __init__(
        self,
        jurors: list[Any],
        method: str,
        aggregation: str | Callable[[list[float]], float] = "mean",
        *,
        weights: list[float] | None = None,
        threshold: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if not jurors:
            raise ValueError("jurors must be a non-empty list.")

        if (
            isinstance(aggregation, str)
            and aggregation not in _BUILTIN_STRATEGIES
        ):
            raise ValueError(
                f"Unknown aggregation strategy {aggregation!r}. "
                f"Choose one of {sorted(_BUILTIN_STRATEGIES)} or pass a callable."
            )

        if aggregation == "weighted_mean":
            if weights is None:
                raise ValueError(
                    "weights must be provided when aggregation='weighted_mean'."
                )
            if len(weights) != len(jurors):
                raise ValueError(
                    f"len(weights)={len(weights)} must equal len(jurors)={len(jurors)}."
                )

        # Validate ALL jurors have the method.
        for i, juror in enumerate(jurors):
            bound = getattr(juror, method, None)
            if bound is None or not callable(bound):
                raise AttributeError(
                    f"Juror {type(juror).__name__!r} at index {i} has no callable method {method!r}."
                )

        self._jurors = jurors
        self._method = method
        self._aggregation = aggregation
        self._weights = weights
        self._threshold = threshold
        self._max_workers = max_workers or len(jurors)

        # Precompute once in __init__ (O(n)) instead of rebuilding per __call__ (O(n²)).
        self._juror_names: list[str] = self._build_juror_names()

        self.__signature__ = inspect.signature(getattr(jurors[0], method))
        self.__name__ = f"jury_{method}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate the same arguments in parallel across all jurors.

        Always returns ``(score, {"reason": ...})``, matching the
        ``_with_cot_reasons`` convention. Per-juror scores and any CoT
        explanations are embedded in the reason string so they appear in
        OTEL spans and the dashboard automatically.
        """
        results: dict[int, tuple[float, str | None]] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._call_juror, juror, args, dict(kwargs)
                ): idx
                for idx, juror in enumerate(self._jurors)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw = future.result()
                    if isinstance(raw, tuple):
                        score = float(raw[0])
                        meta = (
                            raw[1]
                            if len(raw) > 1 and isinstance(raw[1], dict)
                            else {}
                        )
                        reason: str | None = meta.get("reason")
                    else:
                        score = float(raw)
                        reason = None
                    results[idx] = (score, reason)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Juror %r (index %d) failed: %s",
                        self._juror_names[idx],
                        idx,
                        exc,
                    )

        if not results:
            raise RuntimeError(
                f"All {len(self._jurors)} jurors failed to produce a score."
            )

        ordered_idxs = sorted(results.keys())
        scores_by_idx = {idx: results[idx][0] for idx in ordered_idxs}
        agg_score = self._aggregate(scores_by_idx)

        lines = [f"Aggregation: {self._aggregation} → {agg_score:.3f}"]
        for idx in ordered_idxs:
            score, reason = results[idx]
            lines.append(f"  {self._juror_names[idx]}: {score:.3f}")
            if reason:
                for line in reason.splitlines():
                    lines.append(f"    {line}")

        return agg_score, {"reason": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_juror(
        self, juror: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        return getattr(juror, self._method)(*args, **kwargs)

    def _build_juror_names(self) -> list[str]:
        bases = [
            str(getattr(j, "model_engine", None) or type(j).__name__)
            for j in self._jurors
        ]
        return [
            f"{base}[{i}]" if bases.count(base) > 1 else base
            for i, base in enumerate(bases)
        ]

    def _aggregate(self, scores_by_idx: dict[int, float]) -> float:
        ordered = sorted(scores_by_idx.items())
        scores = [s for _, s in ordered]

        if callable(self._aggregation) and not isinstance(
            self._aggregation, str
        ):
            return float(self._aggregation(scores))

        if self._aggregation == "mean":
            return statistics.mean(scores)

        if self._aggregation == "median":
            return statistics.median(scores)

        if self._aggregation == "trimmed_mean":
            if len(scores) < 3:
                return statistics.mean(scores)
            return statistics.mean(sorted(scores)[1:-1])

        if self._aggregation == "majority_vote":
            votes = sum(1 for s in scores if s >= self._threshold)
            if votes * 2 == len(scores):
                logger.warning(
                    "Jury majority_vote tie (%d/%d). Falling back to median.",
                    votes,
                    len(scores),
                )
                return float(statistics.median(scores))
            return float(int(votes > len(scores) / 2))

        if self._aggregation == "weighted_mean":
            total = sum(self._weights[idx] for idx in scores_by_idx)
            if total == 0.0:
                raise ValueError(
                    "weighted_mean: total weight of surviving jurors is 0.0."
                )
            return sum(self._weights[idx] * s for idx, s in ordered) / total

        raise ValueError(f"Unknown aggregation: {self._aggregation!r}")
