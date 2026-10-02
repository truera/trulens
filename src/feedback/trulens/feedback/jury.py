"""LLM Jury — ensemble multiple LLM judges into a single feedback callable.

A single LLM judge is noisy and subject to intra-model bias. Ensembling a
*panel* of diverse judges (a "jury") improves reliability, reduces bias, and
can be cheaper when smaller models are used.

A jury can also repeat *one* judge (see :meth:`Jury.repeated`). At non-zero
temperature the same judge, given identical inputs, does not always return
the same verdict, so repeating it measures run-to-run variance that a single
draw hides.

Every result carries a reliability summary over the juror scores: their
spread, and for a pass/fail view the flip rate and outcome entropy. The
reliability metrics follow "The Coin Flip Judge? Reliability and Bias in
LLM-as-a-Judge Evaluation" (arXiv:2606.13685), which measured pairwise judge
verdicts flipping on average 13.6 percent of the time across repeated
identical evaluations.

"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import inspect
import logging
import math
import statistics
from typing import Any

from trulens.feedback.llm_provider import UNPARSABLE_SCORE

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

    ``__call__`` always returns ``(score, {"reason": ..., "reliability.*":
    ...})``, matching the ``_with_cot_reasons`` convention so per-juror
    breakdowns flow into ``FeedbackCall.meta["reason"]`` and are visible in
    OTEL spans and the dashboard without any UI changes. The reliability keys
    are flat so each one lands on the eval span as its own typed attribute:

    - ``reliability.n_scores``: jurors that returned a score.
    - ``reliability.scores``: those scores, in juror order.
    - ``reliability.score_std``: population standard deviation of the scores.
    - ``reliability.flip_rate``: share of scores that disagree with the
      majority pass/fail verdict at *threshold*, ``1 - max(counts) / n``. This
      is the flip rate defined in arXiv:2606.13685, from 0.0 (unanimous) to
      0.5 (even split).
    - ``reliability.outcome_entropy``: entropy in bits of the pass/fail split,
      from 0.0 (unanimous) to 1.0 (even split).
    - ``reliability.temperature``: the ``temperature`` passed to the jurors,
      or ``None`` when none was passed.

    A juror counts as failed when it raises *or* when its judge reply carried
    no parseable score, which the providers report as
    :data:`~trulens.feedback.llm_provider.UNPARSABLE_SCORE` (``-1.0``). Either
    way it gets no vote. Only a jury with no surviving juror raises.

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
        threshold: Binarisation threshold for ``"majority_vote"`` and for
            the pass/fail view behind ``reliability.flip_rate`` and
            ``reliability.outcome_entropy``. Scores >= *threshold* count as a
            positive vote. Defaults to ``0.5``. On an exact
            ``"majority_vote"`` tie falls back to median.
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

        # Set by ``repeated``: every juror is the same judge, so trials only
        # differ when they are sampled.
        self._repeated = False
        self._warned_unsampled = False

    @classmethod
    def repeated(
        cls,
        judge: Any,
        method: str,
        n_trials: int = 5,
        aggregation: str | Callable[[list[float]], float] = "mean",
        *,
        threshold: float = 0.5,
        max_workers: int | None = None,
    ) -> Jury:
        """Run one judge ``n_trials`` times and aggregate the scores.

        Repeating a single judge measures its run-to-run variance, which a
        single draw hides. Judge methods and ``Metric`` both default to
        ``temperature=0.0``; set a non-zero temperature on the ``Metric`` to
        sample distinct verdicts, and it is passed to every trial. At
        temperature 0 the trials are not sampled, so ``reliability.*`` then
        reflects only nondeterminism in the serving stack. A nonzero flip
        rate is still a real signal, but a zero does not show the judge is
        stable, and the jury logs a warning once. ``reliability.temperature``
        records the temperature the trials ran at.

        With ``n_trials >= 2`` the trials are labeled ``model[0]``,
        ``model[1]``, ... in the reason text. arXiv:2606.13685 found about 11
        trials are needed for a majority verdict to match a 50-trial
        reference with 95 percent probability, so raise ``n_trials`` when the
        verdict matters more than the cost.

        Example::

            from trulens.core import Metric
            from trulens.feedback import Jury
            from trulens.providers.openai import OpenAI

            judge = Jury.repeated(
                OpenAI(model_engine="gpt-4o-mini"), method="relevance", n_trials=5
            )
            metric = Metric(
                implementation=judge, name="Relevance (5 trials)", temperature=0.7
            ).on_input().on_output()
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
        jury = cls(
            jurors=[judge] * n_trials,
            method=method,
            aggregation=aggregation,
            threshold=threshold,
            max_workers=max_workers,
        )
        jury._repeated = True
        return jury

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate the same arguments in parallel across all jurors.

        Always returns ``(score, {"reason": ..., "reliability.*": ...})``,
        matching the ``_with_cot_reasons`` convention. Per-juror scores and
        any CoT explanations are embedded in the reason string so they appear
        in OTEL spans and the dashboard automatically. The ``reliability.*``
        keys are described on the class.
        """
        temperature = kwargs.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        if (
            self._repeated
            and len(self._jurors) > 1
            and not temperature
            and not self._warned_unsampled
        ):
            self._warned_unsampled = True
            logger.warning(
                "Jury %r repeats one judge %d times at temperature %s, so "
                "the trials are not sampled and reliability.* reflects only "
                "serving nondeterminism. A flip rate of 0 here does not show "
                "the judge is stable. Set a non-zero temperature on the "
                "Metric to sample the trials.",
                self.__name__,
                len(self._jurors),
                temperature,
            )

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
                    if score == UNPARSABLE_SCORE:
                        # The judge answered without a score a parser could
                        # find. That is a failure to grade, not a verdict at
                        # the bottom of the scale, so this juror gets no vote
                        # just as if it had raised.
                        logger.warning(
                            "Juror %r (index %d) returned no parsable "
                            "score (%s); its vote is dropped.",
                            self._juror_names[idx],
                            idx,
                            UNPARSABLE_SCORE,
                        )
                        continue
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
        reliability = _reliability_summary(
            [scores_by_idx[idx] for idx in ordered_idxs], self._threshold
        )
        reliability["reliability.temperature"] = temperature

        lines = [
            f"Aggregation: {self._aggregation} → {agg_score:.3f}",
            f"Reliability: {reliability['reliability.n_scores']} scores, "
            f"std {reliability['reliability.score_std']:.3f}, "
            f"flip rate {reliability['reliability.flip_rate']:.3f}, "
            f"entropy {reliability['reliability.outcome_entropy']:.3f}",
        ]
        for idx in ordered_idxs:
            score, reason = results[idx]
            lines.append(f"  {self._juror_names[idx]}: {score:.3f}")
            if reason:
                for line in reason.splitlines():
                    lines.append(f"    {line}")

        return agg_score, {"reason": "\n".join(lines), **reliability}

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


def _reliability_summary(
    scores: list[float], threshold: float
) -> dict[str, Any]:
    """Reliability signals over the surviving juror scores.

    ``flip_rate`` and ``outcome_entropy`` use the pass/fail view of the scores
    (each binarised at *threshold*). ``flip_rate`` is the share of scores
    that disagree with the majority verdict, as defined in arXiv:2606.13685.
    """
    n = len(scores)
    n_pass = sum(1 for s in scores if s >= threshold)
    counts = (n - n_pass, n_pass)

    entropy = 0.0
    for c in counts:
        if c:
            p = c / n
            entropy -= p * math.log2(p)

    return {
        "reliability.n_scores": n,
        "reliability.scores": scores,
        "reliability.score_std": statistics.pstdev(scores) if n > 1 else 0.0,
        "reliability.flip_rate": 1.0 - max(counts) / n,
        "reliability.outcome_entropy": entropy,
    }
