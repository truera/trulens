import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import serial as serial_utils
from trulens.feedback import llm_provider

logger = logging.getLogger(__name__)


class NuggetizedFeedback:
    """
    Wrapper that adds nuggetization on top of standard TruLens feedback functions.

    Uses the provider's built-in nugget extraction and classification methods.
    """

    def __init__(self, provider: llm_provider.LLMProvider):
        """
        Initialize nuggetized feedback wrapper.

        Args:
            provider: TruLens LLM provider with nugget extraction capabilities
        """
        self.provider = provider
        logger.info(
            f"Initialized NuggetizedFeedback with provider: {provider.__class__.__name__}"
        )

    def groundedness_nuggetized(
        self,
        source: str,
        statement: str,
        query: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """
        Evaluate groundedness at nugget level.

        Args:
            source: Source context
            statement: Statement to evaluate
            query: Optional query for nugget extraction context

        Returns:
            Tuple of (aggregate_score, metadata_dict)
        """
        # Extract nuggets from statement using the query (or statement itself) as context
        query_text = query if query is not None else statement
        nuggets = self.provider.extract_nuggets(
            context_text=statement, query_text=query_text
        )
        logger.debug(f"Extracted {len(nuggets)} nuggets from statement")

        if not nuggets:
            return 0.0, {
                "nugget_evaluations": [],
                "total_nuggets": 0,
                "method": "nuggetized",
                "reason": "No nuggets extracted",
            }

        # Classify nuggets for importance weighting
        classifications = self.provider.classify_nuggets(
            nuggets=nuggets, query_text=query_text
        )

        # Evaluate each nugget
        nugget_evaluations = []
        for i, nugget in enumerate(nuggets):
            score, reasons = (
                self.provider.groundedness_measure_with_cot_reasons(
                    source=source, statement=nugget
                )
            )

            importance = (
                classifications[i] if i < len(classifications) else "okay"
            )
            nugget_evaluations.append({
                "nugget": nugget,
                "importance": importance,
                "score": score,
                "reasons": reasons,
            })

        # Aggregate with importance weighting
        aggregate_score = self._aggregate_scores(
            nugget_evaluations, score_key="score"
        )

        metadata = {
            "nugget_evaluations": nugget_evaluations,
            "total_nuggets": len(nuggets),
            "method": "nuggetized",
        }

        return aggregate_score, metadata

    def relevance_nuggetized(
        self, prompt: str, response: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate answer relevance at nugget level.

        Args:
            prompt: Question/prompt
            response: Answer/response to evaluate

        Returns:
            Tuple of (aggregate_score, metadata_dict)
        """
        # Extract nuggets from response
        nuggets = self.provider.extract_nuggets(
            context_text=response, query_text=prompt
        )
        logger.debug(f"Extracted {len(nuggets)} nuggets from response")

        if not nuggets:
            return 0.0, {
                "nugget_evaluations": [],
                "total_nuggets": 0,
                "method": "nuggetized",
                "reason": "No nuggets extracted",
            }

        # Classify nuggets for importance weighting
        classifications = self.provider.classify_nuggets(
            nuggets=nuggets, query_text=prompt
        )

        # Evaluate each nugget
        nugget_evaluations = []
        for i, nugget in enumerate(nuggets):
            score, reasons = self.provider.relevance_with_cot_reasons(
                prompt=prompt, response=nugget
            )

            importance = (
                classifications[i] if i < len(classifications) else "okay"
            )
            nugget_evaluations.append({
                "nugget": nugget,
                "importance": importance,
                "score": score,
                "reasons": reasons,
            })

        # Aggregate with importance weighting
        aggregate_score = self._aggregate_scores(
            nugget_evaluations, score_key="score"
        )

        metadata = {
            "nugget_evaluations": nugget_evaluations,
            "total_nuggets": len(nuggets),
            "method": "nuggetized",
        }

        return aggregate_score, metadata

    def _aggregate_scores(
        self, nugget_evaluations: List[Dict], score_key: str = "score"
    ) -> float:
        """
        Aggregate nugget scores using importance weighting.

        Args:
            nugget_evaluations: List of nugget evaluation dictionaries
            score_key: Key for score in evaluation dict

        Returns:
            Weighted average score (vital nuggets get 2x weight, okay nuggets get 1x weight)
        """
        if not nugget_evaluations:
            return 0.0

        # Calculate weighted sum: vital = 2x, okay = 1x
        total_weight = 0.0
        weighted_sum = 0.0

        for eval_dict in nugget_evaluations:
            importance = eval_dict.get("importance", "okay").lower()
            weight = 2.0 if importance == "vital" else 1.0

            total_weight += weight
            weighted_sum += eval_dict[score_key] * weight

        if total_weight == 0:
            # Fallback to equal weighting
            return np.mean([
                eval_dict[score_key] for eval_dict in nugget_evaluations
            ])

        return weighted_sum / total_weight


def nuggetized_rag_triad(
    provider: llm_provider.LLMProvider,
    question: Optional[serial_utils.Lens] = None,
    answer: Optional[serial_utils.Lens] = None,
    context: Optional[serial_utils.Lens] = None,
    use_nuggets: bool = True,
) -> Dict[str, core_feedback.Feedback]:
    """
    Create a nuggetized RAG triad for granular evaluation.

    This function creates feedback functions that evaluate at the nugget level,
    providing more detailed insights than traditional evaluation.

    Args:
        provider: The LLM provider to use (must have nugget extraction capabilities)
        question: Selector for the question part
        answer: Selector for the answer part
        context: Selector for the context part
        use_nuggets: Whether to use nuggetization (True) or fall back to standard (False)

    Returns:
        Dictionary of feedback functions with nugget-level evaluation
    """

    if not use_nuggets:
        # Fall back to standard RAG triad
        from trulens.feedback.feedback import rag_triad

        return rag_triad(provider, question, answer, context)

    # Initialize nuggetized wrapper using the provider's methods
    nugget_wrapper = NuggetizedFeedback(provider)

    assert hasattr(
        provider, "relevance_with_cot_reasons"
    ), "Need a provider with the `relevance_with_cot_reasons` feedback function."
    assert hasattr(
        provider, "context_relevance_with_cot_reasons"
    ), "Need a provider with the `context_relevance_with_cot_reasons` feedback function."
    assert hasattr(
        provider, "groundedness_measure_with_cot_reasons"
    ), "Need a provider with the `groundedness_measure_with_cot_reasons` feedback function."
    assert hasattr(
        provider, "extract_nuggets"
    ), "Need a provider with the `extract_nuggets` method."
    assert hasattr(
        provider, "classify_nuggets"
    ), "Need a provider with the `classify_nuggets` method."

    are_complete: bool = True
    ret = {}

    # Create nuggetized feedback functions
    for f_imp, f_agg, arg1name, arg1lens, arg2name, arg2lens, f_name in [
        (
            nugget_wrapper.groundedness_nuggetized,
            None,  # No additional aggregation needed - already aggregated
            "source",
            context and context.collect(),
            "statement",
            answer,
            "Groundedness (Nuggetized)",
        ),
        (
            nugget_wrapper.relevance_nuggetized,
            None,  # No additional aggregation needed - already aggregated
            "prompt",
            question,
            "response",
            answer,
            "Answer Relevance (Nuggetized)",
        ),
        (
            # Context relevance doesn't need nuggetization
            provider.context_relevance_with_cot_reasons,
            np.mean,
            "question",
            question,
            "context",
            context,
            "Context Relevance",
        ),
    ]:
        f = core_feedback.Feedback(f_imp, if_exists=context, name=f_name)

        # Only aggregate if specified
        if f_agg is not None:
            f = f.aggregate(f_agg)

        if arg1lens is not None:
            f = f.on(**{arg1name: arg1lens})
        else:
            are_complete = False

        if arg2lens is not None:
            f = f.on(**{arg2name: arg2lens})
        else:
            are_complete = False

        ret[f.name] = f

    if not are_complete:
        logger.warning(
            "Some or all nuggetized RAG triad feedback functions do not have all their selectors set. "
            "This may be ok if they are to be used for colang actions."
        )

    return ret


def compare_traditional_vs_nuggetized(
    provider: llm_provider.LLMProvider,
    question: str,
    answer: str,
    context: str,
) -> Dict:
    """
    Compare traditional vs nuggetized evaluation side-by-side.

    This utility function is useful for testing and demonstrating the
    differences between traditional and nuggetized evaluation.

    Args:
        provider: TruLens LLM provider with nugget extraction capabilities
        question: Question text
        answer: Answer text
        context: Context text

    Returns:
        Dictionary with comparison results
    """
    # Traditional evaluation
    trad_groundedness, trad_g_reasons = (
        provider.groundedness_measure_with_cot_reasons(
            source=context, statement=answer
        )
    )
    trad_relevance, trad_r_reasons = provider.relevance_with_cot_reasons(
        prompt=question, response=answer
    )

    # Nuggetized evaluation
    nugget_wrapper = NuggetizedFeedback(provider)
    nugget_groundedness, nugget_g_metadata = (
        nugget_wrapper.groundedness_nuggetized(
            source=context, statement=answer, query=question
        )
    )
    nugget_relevance, nugget_r_metadata = nugget_wrapper.relevance_nuggetized(
        prompt=question, response=answer
    )

    return {
        "traditional": {
            "groundedness": trad_groundedness,
            "groundedness_reasons": trad_g_reasons,
            "relevance": trad_relevance,
            "relevance_reasons": trad_r_reasons,
        },
        "nuggetized": {
            "groundedness": nugget_groundedness,
            "groundedness_metadata": nugget_g_metadata,
            "relevance": nugget_relevance,
            "relevance_metadata": nugget_r_metadata,
        },
        "comparison": {
            "groundedness_diff": nugget_groundedness - trad_groundedness,
            "relevance_diff": nugget_relevance - trad_relevance,
            "total_nuggets_groundedness": nugget_g_metadata.get(
                "total_nuggets", 0
            ),
            "total_nuggets_relevance": nugget_r_metadata.get(
                "total_nuggets", 0
            ),
        },
    }
