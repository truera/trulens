"""LLM Jury — ensemble multiple LLM judges into a single feedback callable.

A single LLM judge is noisy and subject to intra-model bias. Ensembling a
*panel* of diverse judges (a "jury") improves reliability, reduces bias, and
can be cheaper when smaller models are used.

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import inspect
import logging
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    Args:
        jurors: Non-empty list of ``LLMProvider`` instances.
        method: Name of the feedback method to call on each juror, e.g.
            ``"relevance"`` or ``"groundedness_measure_with_cot_reasons"``.
        aggregation: How to combine individual juror scores. Accepts a
            strategy name (``"mean"``, ``"median"``, ``"trimmed_mean"``,
            ``"majority_vote"``, ``"weighted_mean"``) or any
            ``Callable[[List[float]], float]``. Defaults to ``"mean"``.
        weights: Per-juror weights for ``"weighted_mean"``. Must have the
            same length as *jurors*. When a juror fails its weight is
            redistributed proportionally among the successful ones.
        threshold: Binarisation threshold for ``"majority_vote"``.
            Scores >= *threshold* count as a positive vote. Defaults to
            ``0.5``.
        return_details: When ``True``, ``__call__`` returns a
            ``(score, details)`` tuple where *details* maps each juror's
            model name to its individual score.
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
        jurors: List[Any],
        method: str,
        aggregation: Union[str, Callable[[List[float]], float]] = "mean",
        *,
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
        return_details: bool = False,
        max_workers: Optional[int] = None,
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

        # Validate method exists and capture its signature for Metric introspection.
        bound_method = getattr(jurors[0], method, None)
        if bound_method is None or not callable(bound_method):
            raise AttributeError(
                f"Juror {type(jurors[0]).__name__!r} has no callable method {method!r}."
            )

        self._jurors = jurors
        self._method = method
        self._aggregation = aggregation
        self._weights = weights
        self._threshold = threshold
        self._return_details = return_details
        self._max_workers = max_workers or len(jurors)

        # Expose the provider method's signature so Metric's selector validation
        # works correctly (inspect.signature checks __signature__ first).
        self.__signature__ = inspect.signature(bound_method)
        self.__name__ = f"jury_{method}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, **kwargs: Any
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Evaluate *kwargs* in parallel across all jurors and return an aggregated score.

        Args:
            **kwargs: The same keyword arguments accepted by the underlying
                provider method (e.g. ``prompt`` and ``response`` for
                ``relevance``).

        Returns:
            A float score, or a ``(float, dict)`` tuple when
            *return_details* is ``True``.
        """
        # idx -> score; preserves juror order for weighted_mean.
        results: Dict[int, float] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_juror, juror, kwargs): idx
                for idx, juror in enumerate(self._jurors)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw = future.result()
                    score = (
                        float(raw[0]) if isinstance(raw, tuple) else float(raw)
                    )
                    results[idx] = score
                except Exception as exc:
                    logger.warning(
                        "Juror %r (index %d) failed: %s",
                        self._juror_name(idx),
                        idx,
                        exc,
                    )

        if not results:
            raise RuntimeError(
                f"All {len(self._jurors)} jurors failed to produce a score."
            )

        ordered = sorted(results.items())  # [(idx, score), ...]
        agg_score = self._aggregate(ordered)

        if self._return_details:
            details = {self._juror_name(idx): score for idx, score in ordered}
            return agg_score, details
        return agg_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_juror(self, juror: Any, kwargs: dict) -> Any:
        return getattr(juror, self._method)(**kwargs)

    def _juror_name(self, idx: int) -> str:
        juror = self._jurors[idx]
        engine = getattr(juror, "model_engine", None)
        base = str(engine) if engine else type(juror).__name__
        # Append index only when the same base name appears more than once.
        all_bases = [
            str(getattr(j, "model_engine", None) or type(j).__name__)
            for j in self._jurors
        ]
        return f"{base}[{idx}]" if all_bases.count(base) > 1 else base

    def _aggregate(self, ordered: List[Tuple[int, float]]) -> float:
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
            return float(int(votes > len(scores) / 2))

        if self._aggregation == "weighted_mean":
            idxs = [idx for idx, _ in ordered]
            total = sum(self._weights[i] for i in idxs)
            return sum(self._weights[i] * s for i, s in ordered) / total

        raise ValueError(f"Unknown aggregation: {self._aggregation!r}")
