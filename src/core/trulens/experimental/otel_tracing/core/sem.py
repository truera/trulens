from enum import Enum
from typing import ClassVar, List, Optional, TypeVar

from semconv.trulens.semconv import trace
from trulens.experimental.otel_tracing.core.trace import Span

T = TypeVar("T")


class SpanType(str, Enum):
    TRACE_ROOT = "trace"
    """Spans as collected by tracing system."""

    SEM_ROOT = "sem"
    """Tracing spans converted into "semantic spans", organizing the collected
    spans into more meaningful types and relationships.
    """

    EVAL_ROOT = "eval"
    """Feedback function evaluation span.

    Children of this span will include spans of the other types.
    """


class EvalRoot(Span):
    """Root of feedback function evaluation."""

    ATTR: ClassVar[str] = "eval"
    _prop = Span.attribute_property_factory(base=ATTR)


class Retrieval(Span):
    """A retrieval."""

    _prop = Span.attribute_property

    query_text = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.QUERY_TEXT, str
    )
    """Input text whose related contexts are being retrieved."""

    query_embedding = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.QUERY_EMBEDDING, List[float]
    )
    """Embedding of the input text."""

    distance_type = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.DISTANCE_TYPE, str
    )
    """Distance function used for ranking contexts."""

    num_contexts = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.NUM_CONTEXTS, int
    )
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS, List[str]
    )
    """The retrieved contexts."""

    retrieved_scores = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_SCORES, List[float]
    )
    """The scores of the retrieved contexts."""

    retrieved_embeddings = Span.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_EMBEDDINGS, List[List[float]]
    )
    """The embeddings of the retrieved contexts."""


class Reranking(Span):
    """A reranker call."""

    query_text = Span.attribute_property(
        trace.SpanAttributes.RERANKING.QUERY_TEXT, str
    )
    """The query text."""

    model_name = Span.attribute_property(
        trace.SpanAttributes.RERANKING.MODEL_NAME, str
    )  # consider generic ML model name attr
    """The model name of the reranker."""

    top_n = Span.attribute_property(trace.SpanAttributes.RERANKING.TOP_N, int)
    """The number of contexts to rerank."""

    input_context_texts = Span.attribute_property(
        trace.SpanAttributes.RERANKING.INPUT_CONTEXT_TEXTS, List[str]
    )
    """The contexts being reranked."""

    input_context_scores = Span.attribute_property(
        trace.SpanAttributes.RERANKING.INPUT_CONTEXT_SCORES,
        Optional[List[float]],
    )
    """The scores of the input contexts."""

    output_ranks = Span.attribute_property(
        trace.SpanAttributes.RERANKING.OUTPUT_RANKS, List[int]
    )
    """Reranked indexes into `input_context_texts`."""


class Generation(Span):
    """A generation call to an LLM."""

    model_name = Span.attribute_property(
        trace.SpanAttributes.GENERATION.MODEL_NAME, str
    )  # to replace with otel's LLM_REQUEST_MODEL
    """The model name of the LLM."""

    model_type = Span.attribute_property(
        trace.SpanAttributes.GENERATION.MODEL_TYPE, str
    )
    """The type of model used."""

    input_token_count = Span.attribute_property(
        trace.SpanAttributes.GENERATION.INPUT_TOKEN_COUNT, int
    )  # to replace with otel's LLM_RESPONSE_USAGE_PROMPT_TOKENS
    """The number of tokens in the input."""

    input_messages = Span.attribute_property(
        trace.SpanAttributes.GENERATION.INPUT_MESSAGES, List[dict]
    )
    """The prompt given to the LLM."""

    output_token_count = Span.attribute_property(
        trace.SpanAttributes.GENERATION.OUTPUT_MESSAGES, int
    )  # to replace with otel's LLM_RESPONSE_COMPLETION_TOKENS
    """The number of tokens in the output."""

    output_messages = Span.attribute_property(
        trace.SpanAttributes.GENERATION.OUTPUT_MESSAGES, List[dict]
    )
    """The returned text."""

    temperature = Span.attribute_property(
        trace.SpanAttributes.GENERATION.TEMPERATURE, float
    )  # to replace with otel's LLM_REQUEST_TEMPERATURE
    """The temperature used for generation."""

    cost = Span.attribute_property(trace.SpanAttributes.GENERATION.COST, float)
    """The cost of the generation."""


class Memorization(Span):
    """A memory call."""

    memory_type = Span.attribute_property(
        trace.SpanAttributes.MEMORIZATION.MEMORY_TYPE, str
    )
    """The type of memory."""

    remembered = Span.attribute_property(
        trace.SpanAttributes.MEMORIZATION.REMEMBERED, str
    )
    """The text being integrated into the memory in this span."""


class Embedding(Span):
    """An embedding call."""

    input_text = Span.attribute_property(
        trace.SpanAttributes.EMBEDDING.INPUT_TEXT, str
    )
    """The text being embedded."""

    model_name = Span.attribute_property(
        trace.SpanAttributes.EMBEDDING.MODEL_NAME, str
    )
    """The model name of the embedding model."""

    embedding = Span.attribute_property(
        trace.SpanAttributes.EMBEDDING.EMBEDDING, List[float]
    )
    """The embedding of the input text."""


class ToolInvocation(Span):
    """A tool invocation."""

    ATTR: ClassVar[str] = "tool_invocation"

    description = Span.attribute_property(
        trace.SpanAttributes.TOOLINVOCATION.DESCRIPTION, str
    )
    """The description of the tool."""


class AgentInvocation(Span):
    """An agent invocation."""

    ATTR: ClassVar[str] = "agent_invocation"

    description = Span.attribute_property(
        trace.SpanAttributes.AGENTINVOCATION.DESCRIPTION, str
    )
    """The description of the agent."""
