from trulens.core import TruSession
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.feedback.llm_provider import LLMProvider
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import os

def start_observability():
    session = TruSession()
    return session

def create_evals(provider: LLMProvider = None):
    if provider is None:
        provider = OpenAI(model_engine="gpt-4o")

    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on({
            "source": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            ),
        })
        .on_output()
    )
    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()
        .on_output()
    )

    context_relevance_custom_criteria = """
    When the question plausibly requires multiple different sources to answer, score context relevance based on the following criteria:
    - 0: The context is not relevant to any part of the question.
    - 1: The context is somewhat relevant but not sufficient for answering a portion of the question.
    - 2: The context is sufficient for answering a portion of the question.
    - 3: The context is highly relevant and sufficient for answering the complete question.
    """

    # Context relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(
            provider.context_relevance, name="Context Relevance",
            criteria = context_relevance_custom_criteria,
        )
        .on({
            "question": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
            ),
        })
        .on({
            "context": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            ),
        })
        .aggregate(np.mean)  # choose a different aggregation method if you wish
    )

    return [f_context_relevance, f_groundedness, f_answer_relevance]
