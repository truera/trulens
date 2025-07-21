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

BASE_SCOPE = "ai.observability"


class ResourceAttributes:
    APP_ID = BASE_SCOPE + ".app_id"
    """ID of the app that the span belongs to."""

    APP_NAME = BASE_SCOPE + ".app_name"
    """Fully qualified name of the app that the span belongs to."""

    APP_VERSION = BASE_SCOPE + ".app_version"
    """Name of the version that the span belongs to."""


class SpanAttributes:
    """Names of keys in the attributes field of a span.

    In some cases below, we also include span name or span name prefix.
    """

    SPAN_TYPE = BASE_SCOPE + ".span_type"
    """
    Span type attribute.
    """

    RECORD_ID = BASE_SCOPE + ".record_id"
    """ID of the record that the span belongs to."""

    RUN_NAME = BASE_SCOPE + ".run.name"
    """Name of the run that the span belongs to."""

    INPUT_ID = BASE_SCOPE + ".input_id"
    """ID of the input that the span belongs to."""

    INPUT_RECORDS_COUNT = BASE_SCOPE + ".input_records_count"
    """To track the number of records processed in a run."""

    SPAN_GROUPS = BASE_SCOPE + ".span_groups"
    """List of groups that the span belongs to."""

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

        RECORD_ROOT = "record_root"
        """Spans as collected by tracing system."""

        EVAL_ROOT = "eval_root"
        """Feedback function evaluation root span."""

        EVAL = "eval"
        """Feedback function evaluation span information."""

        RETRIEVAL = "retrieval"
        """A retrieval."""

        GENERATION = "generation"
        """A generation call to an LLM."""

        GRAPH_TASK = "graph_task"
        """A graph task function execution."""

        GRAPH_NODE = "graph_node"
        """A graph node execution."""

    class UNKNOWN:
        """Attributes relevant for spans that could not be categorized otherwise."""

        base = BASE_SCOPE + ".unknown"

    class RECORD_ROOT:
        """Attributes for the root span of a record.

        Includes most fields carried over from
        [trulens.core.schema.base.Record][trulens.core.schema.base.Record].
        """

        base = BASE_SCOPE + ".record_root"

        INPUT = base + ".input"
        """Main input to the app."""

        OUTPUT = base + ".output"
        """Main output of the app."""

        ERROR = base + ".error"
        """Main error of the app.

        Exclusive with main output.
        """

        GROUND_TRUTH_OUTPUT = base + ".ground_truth_output"
        """Ground truth of the record."""

    class EVAL_ROOT:
        """Attributes for the root span of a feedback evaluation.

        Includes most of the fields carried over from
        [trulens.core.schema.feedback.FeedbackResult][trulens.core.schema.feedback.FeedbackResult].
        """

        base = BASE_SCOPE + ".eval_root"

        METRIC_NAME = base + ".metric_name"
        """Name of the feedback definition being evaluated."""

        SPAN_GROUP = base + ".span_group"
        """The span group of the inputs to this metric."""

        ARGS_SPAN_ID = base + ".args_metadata.span_id"
        """
        Mapping of argument name to the ID of the span that provided it. Note
        that this is a scope, and not an attribute by itself.

        E.g. If the function has an argument `x` that came from a span with ID
        "abc", then we would have `ARGS_SPAN_ID + ".x"` with value "abc".
        """

        ARGS_SPAN_ATTRIBUTE = base + ".args_metadata.span_attribute"
        """
        Mapping of argument name to the full span attribute name of the span
        that provided it. Note that this is a scope, and not an attribute by
        itself.

        E.g. If the function has an argument `x` that came directly from the
        span attribute "xyz", then we would have `ARGS_SPAN_ATTRIBUTE + ".x"`
        with value "xyz". If a span attribute was not used directly, then this
        is not set.
        """

        ERROR = base + ".error"
        """Error raised during evaluation."""

        SCORE = base + ".score"
        """Score of the evaluation."""

        HIGHER_IS_BETTER = base + ".higher_is_better"
        """Whether higher is better for this feedback function."""

        EXPLANATION = base + ".explanation"
        """Explanation for the score of the evaluation."""

        METADATA = base + ".metadata"
        """Any metadata of the evaluation."""

    class EVAL:
        """Feedback function evaluation span information."""

        base = BASE_SCOPE + ".eval"

        TARGET_RECORD_ID = base + ".target_record_id"
        """Record id of the record being evaluated."""

        EVAL_ROOT_ID = base + ".eval_root_id"
        """Span id for the EVAL_ROOT span this span is under."""

        METRIC_NAME = base + ".metric_name"
        """Name of the feedback definition being evaluated."""

        CRITERIA = base + ".criteria"
        """Criteria for this sub-step."""

        EXPLANATION = base + ".explanation"
        """Explanation for the score for this sub-step."""

        METADATA = base + ".metadata"
        """Any metadata for this sub-step."""

        SCORE = base + ".score"
        """Score for this sub-step."""

        ERROR = base + ".error"
        """Error raised during this sub-step."""

    class COST:
        """Attributes for spans with a cost."""

        base = BASE_SCOPE + ".cost"

        COST = base + ".cost"
        """Cost of the span."""

        CURRENCY = base + ".cost_currency"
        """Currency of the cost."""

        MODEL = base + ".model"
        """Model used that caused any costs."""

        NUM_TOKENS = base + ".num_tokens"
        """Total tokens processed. """

        NUM_PROMPT_TOKENS = base + ".num_prompt_tokens"
        """Number of prompt tokens supplied."""

        NUM_COMPLETION_TOKENS = base + ".num_completion_tokens"
        """Number of completion tokens generated."""

    class CALL:
        """Instrumented method call attributes."""

        base = BASE_SCOPE + ".call"

        FUNCTION = base + ".function"
        """Name of function being tracked."""

        KWARGS = base + ".kwargs"
        """
        Keyword arguments of the function. This is a scope, and not an
        attribute by itself. E.g. If the function has an argument `x` that had
        a value `1`, then we should have an attribute with key `KWARGS + ".x"`
        with value `1`.
        """

        RETURN = base + ".return"
        """Return value of the function if it executed without error."""

        ERROR = base + ".error"
        """Error raised by the function if it executed with an error.

        Serialized using [str][builtins.str].
        """

    class RETRIEVAL:
        """A retrieval."""

        base = BASE_SCOPE + ".retrieval"

        QUERY_TEXT = base + ".query_text"
        """Input text whose related contexts are being retrieved."""

        NUM_CONTEXTS = base + ".num_contexts"
        """The number of contexts requested, not necessarily retrieved."""

        RETRIEVED_CONTEXTS = base + ".retrieved_contexts"
        """The retrieved contexts."""

    class GENERATION:
        base = BASE_SCOPE + ".generation"

    class GRAPH_TASK:
        """A graph task function execution."""

        base = BASE_SCOPE + ".graph_task"

        TASK_NAME = base + ".task_name"
        """Name of the task function."""

        INPUT_STATE = base + ".input_state"
        """Input state to the task."""

        OUTPUT_STATE = base + ".output_state"
        """Output state from the task."""

        ERROR = base + ".error"
        """Error raised during task execution."""

    class GRAPH_NODE:
        """A graph node execution."""

        base = BASE_SCOPE + ".graph_node"

        NODE_NAME = base + ".node_name"
        """Name of the node."""

        INPUT_STATE = base + ".input_state"
        """Input state to the graph."""

        OUTPUT_STATE = base + ".output_state"
        """Output state from the graph."""

        NODES_EXECUTED = base + ".nodes_executed"
        """List of nodes executed in the graph."""

        ERROR = base + ".error"
        """Error raised during graph execution."""
