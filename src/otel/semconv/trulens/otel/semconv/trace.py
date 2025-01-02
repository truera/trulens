"""Semantic conventions for TruLens spans.

This file should not have any dependencies except possibly other semantic
conventions packages so it can be easily imported by tools that want to read
TruLens data but not use TruLens otherwise.

Relevant links:

- [OTEL Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/).

- [OTEL Trace Semantic
  Conventions](https://opentelemetry.io/docs/specs/semconv/general/trace/).

- [OTEL Semantic Conventions for Generative AI
  Systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
"""

from enum import Enum

from opentelemetry.semconv.resource import (
    ResourceAttributes as otel_ResourceAttributes,
)
from opentelemetry.semconv.trace import SpanAttributes as otel_SpanAttributes


class ResourceAttributes:
    # TODO: Some Span attributes should be moved here.
    pass


class SpanAttributes:
    """Names of keys in the attributes field of a span.

    In some cases below, we also include span name or span name prefix.
    """

    BASE = "trulens."
    """
    Base prefix for the other keys.
    """

    SPAN_TYPE = BASE + "span_type"
    """
    Span type attribute.
    """

    SELECTOR_NAME_KEY = "selector_name"
    """
    Key for the user-defined selector name for the current span.
    Here to help us check both trulens.selector_name and selector_name
    to verify the user attributes and make corrections if necessary.
    """

    SELECTOR_NAME = BASE + SELECTOR_NAME_KEY
    """
    User-defined selector name for the current span.
    """

    RECORD_ID = BASE + "record_id"
    """ID of the record that the span belongs to."""

    APP_ID = BASE + "app_id"
    """ID of the app that the span belongs to."""

    ROOT_SPAN_ID = BASE + "root_span_id"
    """ID of the root span of the record that the span belongs to."""

    class SpanType(str, Enum):
        """Span type attribute values.

        The root types indicate the process that initiating the tracking of
        spans (either app tracing or feedback evaluation). Call indicates a
        method/function call. App indicates a call by an app method. Cost
        indicates a call with cost tracking. All others are semantic types.

        The first three classes are exclusive but the others can be mixed in
        with each other.

        Attributes relevant to each span type follow.
        """

        # Exclusive types.

        UNKNOWN = "unknown"
        """Unknown span type."""

        CUSTOM = "custom"
        """Spans created by the user using otel api."""

        RECORDING = "recording"
        """Span encapsulating a TruLens app recording context."""

        SEMANTIC = "semantic"
        """Recognized span, at least to some degree.

        Must include at least one of the semantic mixable types below as well.
        """

        RECORD_ROOT = "record_root"
        """Spans as collected by tracing system."""

        MAIN = "main"
        """The main span of a record."""

        EVAL_ROOT = "eval_root"
        """Feedback function evaluation span."""

        # Non-semantic mixable types indicate presence of common sets of attributes.

        RECORD = "record"
        """A span in a record."""

        CALL = "call"
        """A function call."""

        COST = "cost"
        """A span with a cost."""

        # Semantic mixable types. A span can have multiple of these types.

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

    class RECORDING:
        """Attributes and span name relevant to the recording span type.

        Note that this span is created every time a TruLens app enters a
        recording context even if it is not called. In such cases, there will be
        a RECORDING span without a RECORD_ROOT.
        """

        base = "trulens.recording"

        SPAN_NAME_PREFIX = base + "."
        """Span name will end with app name."""

        APP_ID = base + ".app_id"
        """Id of the app being recorded."""

    class SEMANTIC:
        """Attributes relevant to all semantic span types."""

        base = "trulens.semantic"

    class UNKNOWN:
        """Attributes relevant for spans that could not be categorized otherwise."""

        base = "trulens.unknown"

    class MAIN:
        """Attributes for the main span of a record."""

        base = "trulens.main"

        SPAN_NAME_PREFIX = base + "."

        MAIN_INPUT = base + ".main_input"
        """Main input to the app."""

        MAIN_OUTPUT = base + ".main_output"
        """Main output of the app."""

        MAIN_ERROR = base + ".main_error"
        """Main error of the app.

        Exclusive with main output.
        """

    class RECORD_ROOT:
        """Attributes for the root span of a record.

        Includes most fields carried over from
        [trulens.core.schema.base.Record][trulens.core.schema.base.Record].
        """

        base = "trulens.record_root"

        SPAN_NAME_PREFIX = base + "."
        """Span name will end with app name."""

        APP_NAME = base + ".app_name"
        """Name of the app for whom this is the root."""

        APP_VERSION = base + ".app_version"
        """Version of the app for whom this is the root."""

        APP_ID = base + ".app_id"

        RECORD_ID = base + ".record_id"

        TOTAL_COST = base + ".total_cost"
        """Total cost of the record.

        Note that child spans might include cost type spans. This is the sum of
        all those costs.
        """

    class EVAL_ROOT:
        """Attributes for the root span of a feedback evaluation.

        Includes most of the fields carried over from
        [trulens.core.schema.feedback.FeedbackResult][trulens.core.schema.feedback.FeedbackResult].
        """

        base = "trulens.eval_root"

        TARGET_RECORD_ID = base + ".target_record_id"
        """Record id of the record being evaluated."""

        TARGET_TRACE_ID = base + ".target_trace_id"
        """Trace id of the root span of the record being evaluated."""

        TARGET_SPAN_ID = base + ".target_span_id"
        """Span id of the root span of the record being evaluated."""

        FEEDBACK_NAME = base + ".feedback_name"
        """Name of the feedback definition being evaluated."""

        FEEDBACK_DEFINITION_ID = base + ".feedback_definition_id"
        """Id of the feedback definition being evaluated."""

        STATUS = base + ".status"
        """Status of the evaluation.

        See [trulens.core.schema.feedback.FeedbackResult.status][trulens.core.schema.feedback.FeedbackResult.status] for values.
        """

        TOTAL_COST = base + ".total_cost"
        """Cost of the evaluation.

        Note that sub spans might contain cost type spans. This is the sum of
        all those costs."""

        ERROR = base + ".error"
        """Error raised during evaluation."""

    class COST:
        """Attributes for spans with a cost."""

        base = "trulens.cost"

        COST = base + ".cost"
        """Cost of the span.

        JSONization of
        [trulens.core.schema.base.Cost][trulens.core.schema.base.Cost].
        """

    class RECORD:
        """Attributes for spans traced as part of a recording."""

        base = "trulens.record"

        APP_IDS = base + ".app_ids"
        """Ids of apps that were tracing this span."""

        RECORD_IDS = base + ".record_ids"
        """Map of app id to record id."""

    class CALL:
        """Instrumented method call attributes."""

        base = "trulens.call"

        SPAN_NAME_PREFIX = base + "."
        """Span name will end with the function name."""

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

        BOUND_ARGUMENTS = base + ".bound_arguments"
        """Bindings of the function if arguments were able to be bound.

        Self is not included. Serialized from
        [trulens.core.utils.pyschema.BoundArguments][trulens.core.utils.pyschema.BoundArguments].
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
