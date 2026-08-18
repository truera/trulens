"""Few-shot example optimizer for TruLens feedback functions.

This module provides :class:`FewShotOptimizer`, a utility for selecting the
best-performing subset of few-shot examples to include in an LLM judge prompt.

Motivation
----------
Feedback functions in TruLens accept an ``examples`` parameter that injects
demonstration examples into the judge's system prompt.  Choosing *which*
examples to include has a large effect on scoring quality, but there is
currently no principled way to pick them.  ``FewShotOptimizer`` fills this gap
by scoring every candidate example against a labeled dataset and returning the
subset that maximises agreement with ground-truth scores.

Typical usage
-------------
::

    from trulens.providers.openai import OpenAI
    from trulens.feedback.optimize import FewShotOptimizer

    provider = OpenAI()

    # Candidate demonstrations use the judge's prompt score scale.
    candidates = [
        ({"prompt": "What is 2+2?", "response": "4"}, 3),
        ({"prompt": "What is the capital?", "response": "Paris"}, 3),
        ({"prompt": "Who wrote Hamlet?", "response": "Einstein"}, 0),
        # … more examples …
    ]

    # A separate held-out dataset used to *evaluate* which examples help most.
    eval_dataset = [
        ({"prompt": "Explain gravity.", "response": "A force."}, 0.8),
        # …
    ]

    optimizer = FewShotOptimizer(
        feedback_fn=provider.relevance,
        candidates=candidates,
        eval_dataset=eval_dataset,
        n_examples=3,
        examples_format="structured",
    )
    result = optimizer.optimize()

    # Standard TruLens provider methods accept the structured list directly.
    provider.relevance(
        prompt="…",
        response="…",
        examples=result.best_examples,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
import logging

logger = logging.getLogger(__name__)

# Type aliases for clarity.
# A single feedback call's keyword arguments, e.g. {"input": "…", "output": "…"}.
FeedbackKwargs = dict[str, str]

# A labeled example contains feedback arguments and a score. Standard TruLens
# providers require an integer on the prompt's output scale; custom callables may
# use normalized floats with the text format.
LabeledExample = tuple[FeedbackKwargs, float | int]

VALID_METRICS = {
    "pearson",
    "spearman",
    "precision",
    "recall",
    "f1",
    "cohens_kappa",
    "accuracy",
    "mae",
}


@dataclass
class OptimizeResult:
    """Outcome of a :meth:`FewShotOptimizer.optimize` run.

    Attributes
    ----------
    best_examples:
        The subset of candidates selected by the optimizer, each paired with
        its ground-truth score.
    correlation:
        Evaluation score (e.g. Pearson correlation or selected metric score)
        achieved on *eval_dataset* when using ``best_examples``. Higher is better.
        ``None`` if fewer than two eval samples were available.
    candidate_scores:
        Mapping from candidate index → metric score achieved when that
        candidate was *included* in the prompt. Useful for debugging.
    metric_name:
        Name of the metric used for optimization (e.g. "pearson", "f1").
    metric_score:
        The metric score achieved by ``best_examples``.
    """

    best_examples: list[LabeledExample]
    correlation: float | None
    candidate_scores: dict[int, float] = field(default_factory=dict)
    metric_name: str = "pearson"
    metric_score: float | None = None

    def __post_init__(self) -> None:
        if self.metric_score is None:
            self.metric_score = self.correlation
        if self.correlation is None:
            self.correlation = self.metric_score


class FewShotOptimizer:
    """Select the best few-shot examples for a TruLens feedback function.

    The optimizer works by:

    1. Iterating over *candidates* one at a time in parallel rounds.
    2. For each candidate, calling *feedback_fn* on every sample in
       *eval_dataset* with that candidate injected as a few-shot example.
    3. Computing the target metric (e.g., Pearson correlation, F1, precision, recall, Cohen's kappa)
       between predicted and ground-truth scores.
    4. Greedily selecting the *n_examples* candidates with the highest
       metric improvement (greedy forward selection).

    Parameters
    ----------
    feedback_fn:
        A callable that accepts the keyword arguments defined in
        *candidates* plus an optional ``examples`` keyword argument.
        It must return a ``float`` in ``[0, 1]``. Must be thread-safe; called
        concurrently when max_workers > 1. Typically a bound method
        on a :class:`trulens.feedback.LLMProvider` subclass, e.g.
        ``provider.relevance``.
    candidates:
        Pool of demonstration examples to select from.  Each entry is a
        ``(feedback_kwargs, score)`` pair. With ``examples_format="structured"``,
        the score must be an integer on the feedback function's prompt scale.
    eval_dataset:
        Held-out labeled examples used to measure how well a candidate set
        helps the judge.  Should be *disjoint* from *candidates* to avoid
        overfitting.
    n_examples:
        Maximum number of examples to include in the final prompt.
        Defaults to ``3``.
    format_sep:
        Separator inserted between examples returned by
        :meth:`format_examples`. Standard TruLens providers use the structured
        examples directly. Defaults to ``"\\n\\n"``.
    examples_format:
        ``"text"`` preserves the original behavior and passes a formatted
        string to custom feedback callables. ``"structured"`` passes the list
        of ``(feedback_kwargs, score)`` tuples expected by standard TruLens
        providers. Defaults to ``"text"``.
    metric:
        Evaluation metric to optimize. Supported metrics: ``"pearson"``,
        ``"spearman"``, ``"precision"``, ``"recall"``, ``"f1"``,
        ``"cohens_kappa"``, ``"accuracy"``, ``"mae"``. Defaults to ``"pearson"``.
    metric_threshold:
        Threshold used for binarizing scores when computing classification
        metrics (precision, recall, f1, cohens_kappa, accuracy). Defaults to ``0.5``.
    max_workers:
        Maximum number of thread workers used to evaluate candidate sets in parallel.
        Defaults to ``None`` (utilizes ThreadPoolExecutor's default).
    """

    def __init__(
        self,
        feedback_fn: Callable[..., float],
        candidates: list[LabeledExample],
        eval_dataset: list[LabeledExample],
        n_examples: int = 3,
        format_sep: str = "\n\n",
        metric: str = "pearson",
        metric_threshold: float = 0.5,
        max_workers: int | None = None,
        examples_format: str = "text",
    ) -> None:
        if not candidates:
            raise ValueError("`candidates` must not be empty.")
        if not eval_dataset:
            raise ValueError("`eval_dataset` must not be empty.")
        if n_examples < 1:
            raise ValueError("`n_examples` must be >= 1.")
        if examples_format not in {"text", "structured"}:
            raise ValueError(
                "`examples_format` must be either 'text' or 'structured'."
            )
        if examples_format == "structured" and any(
            isinstance(score, bool) or not isinstance(score, int)
            for _, score in candidates
        ):
            raise ValueError(
                "Structured examples require integer scores on the feedback "
                "function's prompt scale."
            )

        metric_lower = metric.lower()
        if metric_lower not in VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Supported metrics are: {sorted(VALID_METRICS)}"
            )

        self.feedback_fn = feedback_fn
        self.candidates = candidates
        self.eval_dataset = eval_dataset
        self.n_examples = n_examples
        self.format_sep = format_sep
        self.examples_format = examples_format
        self.metric = metric_lower
        self.metric_threshold = metric_threshold
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizeResult:
        """Run greedy forward selection and return the best example subset.

        Returns
        -------
        OptimizeResult
            Contains the selected examples, overall evaluation score, and
            per-candidate scores.

        Raises
        ------
        RuntimeError
            If *feedback_fn* raises an exception for every candidate on the
            first eval sample (likely a misconfigured provider).
        """
        selected: list[LabeledExample] = []
        remaining = list(enumerate(self.candidates))
        candidate_scores: dict[int, float] = {}

        for round_num in range(min(self.n_examples, len(self.candidates))):
            best_idx: int | None = None
            best_score: float | None = None

            def _evaluate_candidate(
                item: tuple[int, LabeledExample],
            ) -> tuple[int, float | None]:
                orig_idx, candidate = item
                trial_set = selected + [candidate]
                score = self._score_candidate_set(trial_set)
                return orig_idx, score

            if self.max_workers == 1:
                round_results = [
                    _evaluate_candidate(item) for item in remaining
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    round_results = list(
                        executor.map(_evaluate_candidate, remaining)
                    )

            for orig_idx, score in round_results:
                candidate_scores[orig_idx] = (
                    score if score is not None else -1.0
                )

                if score is not None and (
                    best_score is None or score > best_score
                ):
                    best_score = score
                    best_idx = orig_idx

            if best_idx is None:
                logger.warning(
                    "Round %d: no candidate improved metric '%s' — stopping early.",
                    round_num + 1,
                    self.metric,
                )
                break

            chosen = self.candidates[best_idx]
            selected.append(chosen)
            remaining = [(i, c) for i, c in remaining if i != best_idx]
            logger.info(
                "Round %d: selected candidate %d (%s=%.4f).",
                round_num + 1,
                best_idx,
                self.metric,
                best_score,
            )

        final_score = self._score_candidate_set(selected) if selected else None
        return OptimizeResult(
            best_examples=selected,
            correlation=final_score,
            candidate_scores=candidate_scores,
            metric_name=self.metric,
            metric_score=final_score,
        )

    def format_examples(self, examples: list[LabeledExample]) -> str:
        """Serialize examples for custom callables that accept a string.

        Each example is rendered as a bullet showing the input kwargs and the
        expected score, separated by :attr:`format_sep`.

        Parameters
        ----------
        examples:
            Subset of labeled examples to format, typically the output of
            :meth:`optimize`.

        Returns
        -------
        str
            A human-readable representation. Standard TruLens provider methods
            should receive the structured examples instead.
        """
        parts = []
        for i, (kwargs, score) in enumerate(examples, start=1):
            kwargs_str = "\n".join(f"  {k}: {v}" for k, v in kwargs.items())
            parts.append(
                f"Example {i}:\n{kwargs_str}\n  expected_score: {score:.2f}"
            )
        return self.format_sep.join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_candidate_set(
        self,
        candidate_set: list[LabeledExample],
    ) -> float | None:
        """Evaluate *candidate_set* against :attr:`eval_dataset`.

        Calls :attr:`feedback_fn` on every eval sample with *candidate_set*
        injected in the configured format, then computes the selected metric
        between predicted and ground-truth scores.

        Parameters
        ----------
        candidate_set:
            The demonstration examples to inject into the prompt.

        Returns
        -------
        float or None
            Evaluation metric score, or ``None`` if fewer than two eval samples
            produced valid predictions.
        """
        predicted: list[float] = []
        ground_truth: list[float] = []
        examples = (
            candidate_set
            if self.examples_format == "structured"
            else self.format_examples(candidate_set)
        )

        for kwargs, gt_score in self.eval_dataset:
            try:
                score = self.feedback_fn(**kwargs, examples=examples)
                if score is not None:
                    predicted.append(float(score))
                    ground_truth.append(gt_score)
            except Exception:
                logger.warning(
                    "feedback_fn raised an exception for kwargs=%r — skipping.",
                    kwargs,
                    exc_info=True,
                )

        return self._compute_metric(predicted, ground_truth)

    def _compute_metric(
        self,
        predicted: list[float],
        ground_truth: list[float],
    ) -> float | None:
        """Compute the configured evaluation metric between predicted and ground-truth scores."""
        if len(predicted) < 1 or len(ground_truth) != len(predicted):
            return None

        if self.metric == "pearson":
            return self._pearson_correlation(predicted, ground_truth)
        elif self.metric == "spearman":
            return self._spearman_correlation(predicted, ground_truth)
        elif self.metric == "precision":
            return self._precision(
                predicted, ground_truth, self.metric_threshold
            )
        elif self.metric == "recall":
            return self._recall(predicted, ground_truth, self.metric_threshold)
        elif self.metric == "f1":
            return self._f1_score(
                predicted, ground_truth, self.metric_threshold
            )
        elif self.metric == "cohens_kappa":
            return self._cohens_kappa(
                predicted, ground_truth, self.metric_threshold
            )
        elif self.metric == "accuracy":
            return self._accuracy(
                predicted, ground_truth, self.metric_threshold
            )
        elif self.metric == "mae":
            return self._mae_score(predicted, ground_truth)
        return None

    def _pearson_correlation(
        self,
        predicted: list[float],
        ground_truth: list[float],
    ) -> float | None:
        """Compute Pearson *r* between two equal-length lists of floats."""
        n = len(predicted)
        if n < 2 or len(ground_truth) != n:
            return None

        mean_p = sum(predicted) / n
        mean_g = sum(ground_truth) / n

        numerator = sum(
            (p - mean_p) * (g - mean_g)
            for p, g in zip(predicted, ground_truth, strict=False)
        )
        denom_p = sum((p - mean_p) ** 2 for p in predicted) ** 0.5
        denom_g = sum((g - mean_g) ** 2 for g in ground_truth) ** 0.5

        if denom_p == 0 or denom_g == 0:
            return None  # zero variance — correlation undefined

        return numerator / (denom_p * denom_g)

    def _spearman_correlation(
        self,
        predicted: list[float],
        ground_truth: list[float],
    ) -> float | None:
        """Compute Spearman rank correlation coefficient."""
        if len(predicted) < 2:
            return None

        def _rank(vals: list[float]) -> list[float]:
            sorted_indices = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            i = 0
            while i < len(vals):
                j = i
                while j < len(vals):
                    same_val = (
                        vals[sorted_indices[j]] == vals[sorted_indices[i]]
                    )
                    if not same_val:
                        break
                    j += 1
                avg_rank = sum(range(i + 1, j + 1)) / (j - i)
                for k in range(i, j):
                    ranks[sorted_indices[k]] = avg_rank
                i = j
            return ranks

        rank_p = _rank(predicted)
        rank_g = _rank(ground_truth)
        return self._pearson_correlation(rank_p, rank_g)

    def _precision(
        self,
        predicted: list[float],
        ground_truth: list[float],
        threshold: float,
    ) -> float:
        """Compute precision score given binary classification threshold."""
        p_bin = [1 if p >= threshold else 0 for p in predicted]
        g_bin = [1 if g >= threshold else 0 for g in ground_truth]
        tp = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 1 and g == 1
        )
        fp = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 1 and g == 0
        )
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def _recall(
        self,
        predicted: list[float],
        ground_truth: list[float],
        threshold: float,
    ) -> float:
        """Compute recall score given binary classification threshold."""
        p_bin = [1 if p >= threshold else 0 for p in predicted]
        g_bin = [1 if g >= threshold else 0 for g in ground_truth]
        tp = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 1 and g == 1
        )
        fn = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 0 and g == 1
        )
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def _f1_score(
        self,
        predicted: list[float],
        ground_truth: list[float],
        threshold: float,
    ) -> float:
        """Compute F1 score given binary classification threshold."""
        prec = self._precision(predicted, ground_truth, threshold)
        rec = self._recall(predicted, ground_truth, threshold)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def _cohens_kappa(
        self,
        predicted: list[float],
        ground_truth: list[float],
        threshold: float,
    ) -> float | None:
        """Compute Cohen's kappa agreement coefficient."""
        n = len(predicted)
        if n < 1:
            return None
        p_bin = [1 if p >= threshold else 0 for p in predicted]
        g_bin = [1 if g >= threshold else 0 for g in ground_truth]
        tp = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 1 and g == 1
        )
        tn = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 0 and g == 0
        )
        fp = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 1 and g == 0
        )
        fn = sum(
            1 for p, g in zip(p_bin, g_bin, strict=False) if p == 0 and g == 1
        )

        p_o = (tp + tn) / n
        p_pred_1 = (tp + fp) / n
        p_gt_1 = (tp + fn) / n
        p_pred_0 = (tn + fn) / n
        p_gt_0 = (tn + fp) / n
        p_e = (p_pred_1 * p_gt_1) + (p_pred_0 * p_gt_0)

        if 1 - p_e == 0:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1 - p_e)

    def _accuracy(
        self,
        predicted: list[float],
        ground_truth: list[float],
        threshold: float,
    ) -> float:
        """Compute classification accuracy."""
        n = len(predicted)
        if n < 1:
            return 0.0
        correct = sum(
            1
            for p, g in zip(predicted, ground_truth, strict=False)
            if (p >= threshold) == (g >= threshold)
        )
        return correct / n

    def _mae_score(
        self,
        predicted: list[float],
        ground_truth: list[float],
    ) -> float:
        """Compute 1.0 - Mean Absolute Error score (higher is better)."""
        n = len(predicted)
        if n < 1:
            return 0.0
        mae = (
            sum(
                abs(p - g)
                for p, g in zip(predicted, ground_truth, strict=False)
            )
            / n
        )
        return 1.0 - mae
