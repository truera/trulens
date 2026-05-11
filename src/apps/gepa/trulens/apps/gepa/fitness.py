"""GEPA fitness function adapter for TruLens feedback functions.

GEPA (Genetic/Evolutionary Prompt Adaptation) optimizes prompts using
evolutionary algorithms. A *fitness function* is any callable that scores a
candidate prompt as a float in [0, 1].

This module provides:

- :class:`TruLensFitness` — wraps any TruLens feedback callable into the GEPA
  fitness-function interface and optionally logs each evaluation as a
  ``TruVirtual`` record for dashboard visibility.
- :func:`run_evolution` — a simple (μ+λ) evolutionary loop that uses a
  fitness function to iteratively improve a base prompt.
"""

from __future__ import annotations

import logging
import random
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

logger = logging.getLogger(__name__)


class TruLensFitness:
    """Adapts a TruLens feedback callable as a GEPA-compatible fitness function.

    A GEPA fitness function is any callable ``(prompt: str, **kwargs) -> float``.
    ``TruLensFitness`` forwards the call to a TruLens feedback function
    implementation and, when a ``TruVirtual`` recorder is supplied, logs each
    evaluation as a virtual TruLens record so the full optimization trajectory
    is visible in the TruLens dashboard.

    Args:
        feedback_fn: A TruLens feedback function implementation — any callable
            that accepts text keyword arguments and returns a ``float`` or a
            ``(float, dict)`` tuple (the standard TruLens return format).
        input_key: Keyword argument name the feedback function expects for the
            prompt text. Defaults to ``"prompt"``.
        context_key: Optional keyword argument name for a fixed context string.
            Provide alongside ``context`` to evaluate context-dependent metrics
            such as ``context_relevance``.
        context: Fixed context string forwarded to every evaluation call.
        recorder: Optional ``TruVirtual`` recorder instance. When provided,
            each call is logged as a virtual record for audit and visualization.
    """

    def __init__(
        self,
        feedback_fn: Callable,
        *,
        input_key: str = "prompt",
        context_key: Optional[str] = None,
        context: Optional[str] = None,
        recorder: Optional[Any] = None,
    ) -> None:
        self._feedback_fn = feedback_fn
        self._input_key = input_key
        self._context_key = context_key
        self._context = context
        self._recorder = recorder

    def __call__(self, prompt: str, **kwargs: Any) -> float:
        """Evaluate *prompt* and return a fitness score.

        Args:
            prompt: The prompt string to evaluate.
            **kwargs: Additional keyword arguments forwarded to the underlying
                feedback function.

        Returns:
            Float fitness score, typically in [0, 1].
        """
        call_kwargs: dict = {self._input_key: prompt, **kwargs}
        if self._context_key is not None and self._context is not None:
            call_kwargs[self._context_key] = self._context

        result = self._feedback_fn(**call_kwargs)

        score = float(result[0]) if isinstance(result, tuple) else float(result)

        if self._recorder is not None:
            self._log_record(prompt=prompt, score=score)

        return score

    def _log_record(self, prompt: str, score: float) -> None:
        """Log an evaluation as a TruLens virtual record."""
        try:
            from trulens.apps.virtual import VirtualRecord
            from trulens.core import Select

            call_selector = Select.RecordCalls.fitness_fn.evaluate
            record = VirtualRecord(
                main_input=prompt,
                main_output=str(score),
                calls={
                    call_selector: {
                        "args": [prompt],
                        "rets": score,
                    }
                },
            )
            self._recorder.add_record(record)
        except Exception as exc:
            logger.warning(
                "Failed to log GEPA evaluation to TruLens: %s", exc
            )


def run_evolution(
    base_prompt: str,
    fitness_fn: Callable[[str], float],
    mutate_fn: Callable[[str], str],
    *,
    n_generations: int = 10,
    population_size: int = 5,
    top_k: int = 2,
    seed: Optional[int] = None,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Run a (μ+λ) evolutionary loop to optimize a prompt.

    At each generation:

    1. Score all candidates with *fitness_fn*.
    2. Keep the top-``top_k`` candidates (elitist selection).
    3. Fill the next population by mutating survivors at random.

    When *fitness_fn* is a :class:`TruLensFitness` with a recorder attached,
    every prompt evaluation is logged as a TruLens virtual record, giving a
    generation-by-generation audit trail in the dashboard.

    Args:
        base_prompt: The starting prompt for the evolutionary search.
        fitness_fn: Any callable ``(prompt: str) -> float`` that scores a
            prompt. Use :class:`TruLensFitness` to wrap a TruLens feedback
            function.
        mutate_fn: Callable that takes a prompt string and returns a mutated
            variant (e.g. rephrasing, adding instructions, changing tone).
        n_generations: Number of evolutionary generations to run.
        population_size: Number of candidate prompts evaluated per generation.
        top_k: Number of top-scoring candidates carried over to the next
            generation (elitist survivors).
        seed: Optional random seed for reproducibility.

    Returns:
        A three-tuple ``(best_prompt, best_score, history)`` where *history*
        is a list of ``(best_prompt_in_generation, best_score_in_generation)``
        pairs, one entry per generation.
    """
    if top_k >= population_size:
        raise ValueError(
            f"top_k ({top_k}) must be less than population_size ({population_size})"
        )

    if seed is not None:
        random.seed(seed)

    population: List[str] = [base_prompt] + [
        mutate_fn(base_prompt) for _ in range(population_size - 1)
    ]

    history: List[Tuple[str, float]] = []

    for generation in range(n_generations):
        scored: List[Tuple[str, float]] = [
            (prompt, fitness_fn(prompt)) for prompt in population
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_prompt, best_score = scored[0]
        history.append((best_prompt, best_score))
        logger.info(
            "Generation %d/%d — best score: %.4f",
            generation + 1,
            n_generations,
            best_score,
        )

        survivors = [p for p, _ in scored[:top_k]]
        population = list(survivors)
        while len(population) < population_size:
            parent = random.choice(survivors)
            population.append(mutate_fn(parent))

    best_prompt, best_score = max(history, key=lambda x: x[1])
    return best_prompt, best_score, history
