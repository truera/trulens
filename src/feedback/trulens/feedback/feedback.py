import logging
from typing import Dict, Optional

import numpy as np
from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import serial as serial_utils
from trulens.feedback import llm_provider

logger = logging.getLogger(__name__)


def rag_triad(
    provider: llm_provider.LLMProvider,
    question: Optional[serial_utils.Lens] = None,
    answer: Optional[serial_utils.Lens] = None,
    context: Optional[serial_utils.Lens] = None,
) -> Dict[str, core_feedback.Feedback]:
    """Create a triad of feedback functions for evaluating context retrieval
    generation steps.

    If a particular lens is not provided, the relevant selectors will be
    missing. These can be filled in later or the triad can be used for rails
    feedback actions which fill in the selectors based on specification from
    within colang.

    Args:
        provider: The provider to use for implementing the feedback functions.

        question: Selector for the question part.

        answer: Selector for the answer part.

        context: Selector for the context part.
    """

    assert hasattr(
        provider, "relevance"
    ), "Need a provider with the `relevance` feedback function."
    assert hasattr(
        provider, "context_relevance"
    ), "Need a provider with the `context_relevance` feedback function."

    are_complete: bool = True

    ret = {}

    for f_imp, f_agg, arg1name, arg1lens, arg2name, arg2lens, f_name in [
        (
            provider.groundedness_measure_with_cot_reasons,
            np.mean,
            "source",
            context and context.collect(),
            "statement",
            answer,
            "Groundedness",
        ),
        (
            provider.relevance_with_cot_reasons,
            np.mean,
            "prompt",
            question,
            "response",
            answer,
            "Answer Relevance",
        ),
        (
            provider.context_relevance_with_cot_reasons,
            np.mean,
            "question",
            question,
            "context",
            context,
            "Context Relevance",
        ),
    ]:
        f = core_feedback.Feedback(
            f_imp, if_exists=context, name=f_name
        ).aggregate(f_agg)
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
            "Some or all RAG triad feedback functions do not have all their selectors set. "
            "This may be ok if they are to be used for colang actions."
        )

    return ret
