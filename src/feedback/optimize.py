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

    # A pool of candidate demonstrations: each entry is a
    # (feedback_kwargs, ground_truth_score) pair.
    candidates = [
        ({"input": "What is 2+2?", "output": "4"},          1.0),
        ({"input": "What is the capital?", "output": "Paris"}, 0.9),
        ({"input": "Who wrote Hamlet?", "output": "Einstein"}, 0.1),
        # … more examples …
    ]

    # A separate held-out dataset used to *evaluate* which examples help most.
    eval_dataset = [
        ({"input": "Explain gravity.", "output": "A force."}, 0.8),
        # …
    ]

    optimizer = FewShotOptimizer(
        feedback_fn=provider.relevance,
        candidates=candidates,
        eval_dataset=eval_dataset,
        n_examples=3,
    )
    best_examples = optimizer.optimize()

    # Use the optimized examples with your feedback function.
    provider.relevance(
        input="…",
        output="…",
        examples=optimizer.format_examples(best_examples),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type aliases for clarity.
# A single feedback call's keyword arguments, e.g. {"input": "…", "output": "…"}.
FeedbackKwargs = dict[str, str]

# A labeled example: (feedback_kwargs, ground_truth_score ∈ [0, 1]).
LabeledExample = Tuple[FeedbackKwargs, float]


@dataclass
class OptimizeResult:
    """Outcome of a :meth:`FewShotOptimizer.optimize` run.

    Attributes
    ----------
    best_examples:
        The subset of candidates selected by the optimizer, each paired with
        its ground-truth score.
    correlation:
        Pearson correlation between the feedback function's predicted scores
        and the ground-truth scores on *eval_dataset* when using
        ``best_examples``.  Higher is better.  ``None`` if fewer than two
        eval samples were available.
    candidate_scores:
        Mapping from candidate index → correlation achieved when that
        candidate was *included* in the prompt.  Useful for debugging.
    """

    best_examples: List[LabeledExample]
    correlation: Optional[float]
    candidate_scores: dict[int, float] = field(default_factory=dict)


class FewShotOptimizer:
    """Select the best few-shot examples for a TruLens feedback function.

    The optimizer works by:

    1. Iterating over *candidates* one at a time (or in small batches).
    2. For each candidate, calling *feedback_fn* on every sample in
       *eval_dataset* with that candidate injected as a few-shot example.
    3. Computing the Pearson correlation between predicted and ground-truth
       scores.
    4. Greedily selecting the *n_examples* candidates with the highest
       correlation improvement (greedy forward selection).

    Parameters
    ----------
    feedback_fn:
        A callable that accepts the keyword arguments defined in
        *candidates* plus an optional ``examples: str`` keyword argument.
        It must return a ``float`` in ``[0, 1]``.  Typically a bound method
        on a :class:`trulens.feedback.LLMProvider` subclass, e.g.
        ``provider.relevance``.
    candidates:
        Pool of demonstration examples to select from.  Each entry is a
        ``(feedback_kwargs, ground_truth_score)`` pair where
        ``ground_truth_score`` is a float in ``[0, 1]``.
    eval_dataset:
        Held-out labeled examples used to measure how well a candidate set
        helps the judge.  Should be *disjoint* from *candidates* to avoid
        overfitting.
    n_examples:
        Maximum number of examples to include in the final prompt.
        Defaults to ``3``.
    format_sep:
        Separator inserted between formatted examples when building the
        ``examples`` string passed to *feedback_fn*.  Defaults to ``"\\n\\n"``.
    """

    def __init__(
        self,
        feedback_fn: Callable[..., float],
        candidates: List[LabeledExample],
        eval_dataset: List[LabeledExample],
        n_examples: int = 3,
        format_sep: str = "\n\n",
    ) -> None:
        if not candidates:
            raise ValueError("`candidates` must not be empty.")
        if not eval_dataset:
            raise ValueError("`eval_dataset` must not be empty.")
        if n_examples < 1:
            raise ValueError("`n_examples` must be >= 1.")

        self.feedback_fn = feedback_fn
        self.candidates = candidates
        self.eval_dataset = eval_dataset
        self.n_examples = n_examples
        self.format_sep = format_sep

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizeResult:
        """Run greedy forward selection and return the best example subset.

        Returns
        -------
        OptimizeResult
            Contains the selected examples, overall correlation, and
            per-candidate correlation scores.

        Raises
        ------
        RuntimeError
            If *feedback_fn* raises an exception for every candidate on the
            first eval sample (likely a misconfigured provider).
        """
        raise NotImplementedError(
            "optimize() will be implemented in the next commit."
        )

    def format_examples(self, examples: List[LabeledExample]) -> str:
        """Serialise a list of labeled examples into the string format expected
        by *feedback_fn*'s ``examples`` parameter.

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
            A human-readable string ready to be passed as
            ``feedback_fn(..., examples=<return_value>)``.
        """
        raise NotImplementedError(
            "format_examples() will be implemented in the next commit."
        )

    # ------------------------------------------------------------------
    # Private helpers (stubs — implementations in next commit)
    # ------------------------------------------------------------------

    def _score_candidate_set(
        self,
        candidate_set: List[LabeledExample],
    ) -> Optional[float]:
        """Evaluate *candidate_set* against :attr:`eval_dataset`.

        Calls :attr:`feedback_fn` on every eval sample with the formatted
        *candidate_set* injected as few-shot examples, then computes the
        Pearson correlation between predicted and ground-truth scores.

        Parameters
        ----------
        candidate_set:
            The demonstration examples to inject into the prompt.

        Returns
        -------
        float or None
            Pearson correlation, or ``None`` if fewer than two eval samples
            produced valid predictions.
        """
        raise NotImplementedError

    def _pearson_correlation(
        self,
        predicted: List[float],
        ground_truth: List[float],
    ) -> Optional[float]:
        """Compute Pearson *r* between two equal-length lists of floats.

        Returns ``None`` when the lists have fewer than two elements or when
        one of the lists has zero variance (correlation is undefined).
        """
        raise NotImplementedError
