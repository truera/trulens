"""Semantic conventions for TruLens spans.

This file should not have any dependencies so it can be easily imported by tools
that want to read TruLens data but not use TruLens otherwise.
"""

from enum import Enum

from opentelemetry.semconv.resource import (
    ResourceAttributes as otel_ResourceAttributes,
)
from opentelemetry.semconv.trace import SpanAttributes as otel_SpanAttributes


class ResourceAttributes:
    pass


class SpanAttributes:
    SPAN_TYPE = "trulens.span_type"

    class SpanType(str, Enum):
        """Span type attribute values.

        The root types indicate the process that initiating the tracking of
        spans (either app tracing or feedback evaluation). Call indicates a
        method/function call whereas the other types are semantic app steps. A
        span can be multiple of these types at once except for the roots and
        unknown.
        """

        UNKNOWN = "unknown"
        """Unknown span type."""

        TRACE_ROOT = "trace_root"
        """Spans as collected by tracing system."""

        EVAL_ROOT = "eval_root"
        """Feedback function evaluation span."""

        CALL = "call"
        """A function call."""

        RETRIEVAL = "retrieval"
        """A retrieval."""

        RERANKING = "reranking"
        """A reranker call."""

        GENERATION = "generation"
        """A generation call to an LLM."""

        MEMORIZATION = "memorization"
        """A memory call."""

        EMBEDDING = "embedding"
        """An embedding call."""

        TOOL_INVOCATION = "tool_invocation"
        """A tool invocation."""

        AGENT_INVOCATION = "agent_invocation"
        """An agent invocation."""

    class UNKNOWN:
        base = "trulens.unknown"

    class TRACE_ROOT:
        """Attributes for the root span of a trace."""

        base = "trulens.record"

    class EVAL_ROOT:
        """Attributes for the root span of a feedback evaluation."""

        base = "trulens.eval"

    class CALL:
        """Instrumented method call attributes."""

        base = "trulens.call"

        CALL_ID = base + ".call_id"
        """Unique identifier for the call."""

        STACK = base + ".stack"
        """Call stack."""

        SIGNATURE = base + ".signature"
        """Signature of the function being tracked.

        Serialization of[inspect.Signature][inspect.Signature]."""

        FUNCTION = base + ".function"
        """Function being tracked.

        Serialized from
        [trulens.core.utils.pyschema.FunctionOrMethod][trulens.core.utils.pyschema.FunctionOrMethod]."""

        CLASS = base + ".class"
        """Class owning this function if it is a method.

        Serialized from
        [trulens.core.utils.pyschema.Class][trulens.core.utils.pyschema.Class]."""

        ARGS = base + ".args"
        """Arguments of the function.

        Serialized using
        [trulens.core.utils.json.jsonify][trulens.core.utils.json.jsonify]. If
        the function was a method, self will NOT be included in this list.
        """

        KWARGS = base + ".kwargs"
        """Keyword arguments of the function.

        Serialized using [trulens.core.utils.json.jsonify][trulens.core.utils.json.jsonify].
        """

        BINDINGS = base + ".bindings"
        """Bindings of the function if arguments were able to be bound.

        Serialized from [trulens.core.utils.pyschema.Bindings][trulens.core.utils.pyschema.Bindings].
        """

        RETURN = base + ".return"
        """Return value of the function if it executed without error.

        Serialized using [trulens.core.utils.json.jsonify][trulens.core.utils.json.jsonify].
        """

        ERROR = base + ".error"
        """Error raised by the function if it executed with an error.

        Serialized using [str][builtins.str].
        """

        # TODO: move to ResourceAttributes
        PROCESS_ID = otel_ResourceAttributes.PROCESS_PID
        """Process ID.

        Integer.
        """

        THREAD_ID = otel_SpanAttributes.THREAD_ID
        """Thread ID.

        Integer.
        """

    class RETRIEVAL:
        """A retrieval."""

        base = "trulens.sem.retrieval"

        QUERY_TEXT = base + ".query_text"
        """Input text whose related contexts are being retrieved."""

        QUERY_EMBEDDING = base + ".query_embedding"
        """Embedding of the input text."""

        DISTANCE_TYPE = base + ".distance_type"
        """Distance function used for ranking contexts."""

        NUM_CONTEXTS = base + ".num_contexts"
        """The number of contexts requested, not necessarily retrieved."""

        RETRIEVED_CONTEXTS = base + ".retrieved_contexts"
        """The retrieved contexts."""

        RETRIEVED_SCORES = base + ".retrieved_scores"
        """The scores of the retrieved contexts."""

        RETRIEVED_EMBEDDINGS = base + ".retrieved_embeddings"
        """The embeddings of the retrieved contexts."""

    class RERANKING:
        """A reranker call."""

        base = "trulens.sem.reranking"

        QUERY_TEXT = base + ".query_text"
        """The query text."""

        MODEL_NAME = base + ".model_name"
        """The model name of the reranker."""

        TOP_N = base + ".top_n"
        """The number of contexts to rerank."""

        INPUT_CONTEXT_TEXTS = base + ".input_context_texts"
        """The contexts being reranked."""

        INPUT_CONTEXT_SCORES = base + ".input_context_scores"
        """The scores of the input contexts."""

        OUTPUT_RANKS = base + ".output_ranks"
        """Reranked indexes into `input_context_texts`."""

    class GENERATION:
        base = "trulens.sem.generation"

        # GEN_AI_*

        MODEL_NAME = base + ".model_name"
        """The model name of the LLM."""
        # GEN_AI_REQUEST_MODEL
        # GEN_AI_RESPONSE_MODEL ?

        MODEL_TYPE = base + ".model_type"
        """The type of model used."""

        INPUT_TOKEN_COUNT = base + ".input_token_count"
        """The number of tokens in the input."""
        # GEN_AI_USAGE_INPUT_TOKENS

        INPUT_MESSAGES = base + ".input_messages"
        """The prompt given to the LLM."""
        # GEN_AI_PROMPT

        OUTPUT_TOKEN_COUNT = base + ".output_token_count"
        """The number of tokens in the output."""
        # GEN_AI_USAGE_OUTPUT_TOKENS

        OUTPUT_MESSAGES = base + ".output_messages"
        """The returned text."""

        TEMPERATURE = base + ".temperature"
        """The temperature used for generation."""
        # GEN_AI_REQUEST_TEMPERATURE

        COST = base + ".cost"
        """The cost of the generation."""

    class MEMORIZATION:
        """A memory saving call."""

        base = "trulens.sem.memorization"

        MEMORY_TYPE = base + ".memory_type"
        """The type of memory."""

        REMEMBERED = base + ".remembered"
        """The text being integrated into the memory in this span."""

    class EMBEDDING:
        """An embedding call."""

        base = "trulens.sem.embedding"

        INPUT_TEXT = base + ".input_text"
        """The text being embedded."""

        MODEL_NAME = base + ".model_name"
        """The model name of the embedding model."""

        EMBEDDING = base + ".embedding"
        """The embedding of the input text."""

    class TOOL_INVOCATION:
        """A tool invocation."""

        base = "trulens.sem.tool_invocation"

        DESCRIPTION = base + ".description"
        """The description of the tool."""

    class AGENT_INVOCATION:
        """An agent invocation."""

        base = "trulens.sem.agent_invocation"

        DESCRIPTION = base + ".description"
        """The description of the agent."""
