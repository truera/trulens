"""SelfConsistency — intra-judge reliability wrapper.

A single LLM judge called once is noisy: at default temperature settings the
same prompt does not always return the same score. ``SelfConsistency`` fixes
this by running one judge *N* times with identical inputs and aggregating the
results, giving you a more stable signal without swapping out your provider.

It complements :class:`~trulens.feedback.jury.Jury` (inter-judge diversity):
use ``Jury`` to reduce inter-model bias; use ``SelfConsistency`` to reduce
intra-run variance for a single model.

Because ``SelfConsistency.__call__`` exposes the same parameter names as the
underlying provider method, it plugs directly into
``Metric(implementation=sc)`` — no changes to ``Metric``, ``Selector``, or
the evaluation pipeline are needed.

``__call__`` always returns ``(score, {"reason": ...})``, matching the
``_with_cot_reasons`` convention so per-trial scores, standard deviation,
flip rate, and entropy flow into ``FeedbackCall.meta["reason"]`` and are
visible in OTEL spans and the dashboard without any UI changes.

A trial counts as failed when it raises *or* when its judge reply carried no
parseable score, which the providers report as
:data:`~trulens.feedback.llm_provider.UNPARSABLE_SCORE` (``-1.0``). Either
way it gets no vote. Only a wrapper with no surviving trial raises.

Example::

    from trulens.core import Metric
    from trulens.feedback.self_consistency import SelfConsistency
    from trulens.providers.openai import OpenAI

    sc = SelfConsistency(
        provider=OpenAI(model_engine="gpt-4o-mini"),
        method="relevance",
        n=5,
        aggregation="median",
    )
    m = Metric(implementation=sc, name="SC Relevance").on_input().on_output()
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

from trulens.feedback import llm_provider

logger = logging.getLogger(__name__)

_BUILTIN_STRATEGIES = frozenset({
    "mean",
    "median",
    "trimmed_mean",
    "majority_vote",
})


class SelfConsistency:
    """Run one LLM judge N times to produce a more reliable score.

    ``SelfConsistency`` wraps a single provider instance, calls the same named
    method *n* times in parallel with identical inputs, and aggregates the
    results using a configurable strategy. The structured metadata returned by
    ``__call__`` tracks per-trial scores, standard deviation, flip rate, and
    outcome entropy so reliability can be monitored over time.

    Args:
        provider: An ``LLMProvider`` instance (e.g. ``OpenAI()``,
            ``LiteLLM()``).
        method: Name of the feedback method to call, e.g. ``"relevance"`` or
            ``"groundedness_measure_with_cot_reasons"``.
        n: Number of independent trials to run. Must be >= 2.
        aggregation: How to combine individual trial scores. Accepts a
            strategy name (``"mean"``, ``"median"``, ``"trimmed_mean"``,
            ``"majority_vote"``) or any ``Callable[[list[float]], float]``.
            Defaults to ``"mean"``.
        threshold: Binarisation threshold for ``"majority_vote"`` and for
            computing the flip rate and entropy reliability metrics. Scores
            >= *threshold* count as a positive vote. On an exact tie
            ``"majority_vote"`` falls back to median. Defaults to ``0.5``.
        max_workers: Maximum parallel threads. Defaults to *n* (all trials
            at once).

    Example::

        from trulens.core import Metric
        from trulens.feedback.self_consistency import SelfConsistency
        from trulens.providers.openai import OpenAI

        sc = SelfConsistency(
            provider=OpenAI(model_engine="gpt-4o-mini"),
            method="relevance",
            n=5,
            aggregation="median",
        )
        m = Metric(implementation=sc, name="SC Relevance").on_input().on_output()
    """

    def __init__(
        self,
        provider: Any,
        method: str,
        n: int = 5,
        aggregation: str | Callable[[list[float]], float] = "mean",
        *,
        threshold: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if n < 2:
            raise ValueError(
                f"`n` must be >= 2 to aggregate repeated trials; got {n}."
            )

        bound = getattr(provider, method, None)
        if bound is None or not callable(bound):
            raise AttributeError(
                f"Provider {type(provider).__name__!r} has no callable "
                f"method {method!r}."
            )

        if (
            isinstance(aggregation, str)
            and aggregation not in _BUILTIN_STRATEGIES
        ):
            raise ValueError(
                f"Unknown aggregation strategy {aggregation!r}. "
                f"Choose one of {sorted(_BUILTIN_STRATEGIES)} or pass a callable."
            )

        self._provider = provider
        self._method = method
        self._n = n
        self._aggregation = aggregation
        self._threshold = threshold
        self._max_workers = max_workers or n

        self.__signature__ = inspect.signature(getattr(provider, method))
        self.__name__ = f"self_consistency_{method}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate the same arguments across N independent trials in parallel.

        Always returns ``(score, {"reason": ...})``, matching the
        ``_with_cot_reasons`` convention. Per-trial scores, standard
        deviation, flip rate, and entropy are embedded in the reason string
        so they appear in OTEL spans and the dashboard automatically.
        """
        # results[idx] = (score, reason) keyed by submission index so the
        # reason string labels trials in submission order, not completion order.
        results: dict[int, tuple[float, str | None]] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_trial, args, dict(kwargs)): idx
                for idx in range(self._n)
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
                    if score == llm_provider.UNPARSABLE_SCORE:
                        logger.warning(
                            "SelfConsistency(%r, n=%d): a trial returned no "
                            "parsable score (%s); it is dropped.",
                            self._method,
                            self._n,
                            llm_provider.UNPARSABLE_SCORE,
                        )
                        continue
                    results[idx] = (score, reason)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "SelfConsistency(%r, n=%d): a trial failed: %s",
                        self._method,
                        self._n,
                        exc,
                    )

        if not results:
            raise RuntimeError(
                f"All {self._n} trials of SelfConsistency({self._method!r}) "
                "failed to produce a score."
            )

        ordered_idxs = sorted(results.keys())
        scores = [results[idx][0] for idx in ordered_idxs]

        agg_score = self._aggregate(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        flip_rate = self._flip_rate(scores)
        entropy = self._binary_entropy(scores, threshold=self._threshold)

        lines = [
            f"SelfConsistency({self._method}, n={self._n}, "
            f"agg={self._aggregation!r}) → {agg_score:.3f}",
            f"  std_dev={std_dev:.3f}  flip_rate={flip_rate:.3f}  "
            f"entropy={entropy:.3f}",
            f"  trials={[f'{s:.3f}' for s in scores]}",
        ]
        for idx in ordered_idxs:
            _, r = results[idx]
            if r:
                for line in r.splitlines():
                    lines.append(f"  trial {idx + 1}: {line}")

        return agg_score, {"reason": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_trial(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return getattr(self._provider, self._method)(*args, **kwargs)

    def _aggregate(self, scores: list[float]) -> float:
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
                    "SelfConsistency majority_vote tie (%d/%d). "
                    "Falling back to median.",
                    votes,
                    len(scores),
                )
                return float(statistics.median(scores))
            return float(int(votes > len(scores) / 2))

        raise ValueError(f"Unknown aggregation: {self._aggregation!r}")

    def _flip_rate(self, scores: list[float]) -> float:
        """Fraction of trials that disagreed with the majority verdict.

        Computed as ``1 − (count_of_most_common_verdict / n_trials)``.
        A flip rate of 0.0 means every trial agreed; 0.5 means the judge is
        effectively random.
        """
        if len(scores) <= 1:
            return 0.0
        positives = sum(1 for s in scores if s >= self._threshold)
        negatives = len(scores) - positives
        majority_count = max(positives, negatives)
        return 1.0 - majority_count / len(scores)

    @staticmethod
    def _binary_entropy(scores: list[float], *, threshold: float) -> float:
        """Binary entropy of the positive-vote proportion across trials.

        Returns 0.0 when all trials agree; approaches 1.0 when trials are
        split 50/50.
        """
        if not scores:
            return 0.0
        p = sum(1 for s in scores if s >= threshold) / len(scores)
        if p in (0.0, 1.0):
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
