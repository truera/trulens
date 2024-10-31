from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)

import pydantic
from trulens.experimental.otel_tracing.core.trace import Span
from trulens.semconv import trace

T = TypeVar("T")


class SpanType(str, Enum):
    """Span types.

    The root types indicate the process that initiating the tracking of spans
    (either app tracing or feedback evaluation) whereas the other types are
    semantic app steps.
    """

    UNKNOWN = "unknown"
    """Unknown span type."""

    TRACE_ROOT = "trace"
    """Spans as collected by tracing system."""

    EVAL_ROOT = "eval"
    """Feedback function evaluation span.

    Should include a TRACE_ROOT span as a child.
    """

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


class AttributeProperty(property, Generic[T]):
    """Property that stores its value in the attributes dictionary.

    Validates default and on assignment. This is meant to be used only in
    Span instances (or subclasses).

        Args:
            name: The name of the property. The key used for storage will be
                this with the vendor prefix.

            typ: The type of the property.

            typ_factory: A factory function that returns the type of the
                property. This can be used for forward referenced types.

            default: The default value of the property.

            default_factory: A factory function that returns the default value
                of the property. This can be used for defaults that make use of
                forward referenced types.
    """

    def __init__(
        self,
        name: str,
        typ: Optional[Type[T]] = None,
        typ_factory: Optional[Callable[[], Type[T]]] = None,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None,
    ):
        self.name = name
        self.typ = typ
        self.typ_factory = typ_factory
        self.default = default
        self.default_factory = default_factory

        self.forward_initialized = False

    def init_forward(self):
        if self.forward_initialized:
            return

        self.forward_initialized = True

        if self.typ is None and self.typ_factory is not None:
            self.typ = self.typ_factory()

        if self.default is None and self.default_factory is not None:
            self.default = self.default_factory()

        if self.typ is None and self.default is not None:
            self.typ = type(self.default)

        if self.typ is None:
            self.tadapter = None
        else:
            self.tadapter = pydantic.TypeAdapter(self.typ)
            if self.default is not None:
                self.tadapter.validate_python(self.default)

    def __get__(self, obj: Any, objtype: Optional[Type[T]]) -> Optional[T]:  # type: ignore # noqa: F821
        if obj is None:
            return self

        self.init_forward()
        return obj.attributes.get(self.name, self.default)

    def __set__(self, obj, value: T) -> None:
        self.init_forward()

        if self.tadapter is not None:
            self.tadapter.validate_python(value)

        obj.attributes[self.name] = value

    def __delete__(self, obj):
        del obj.attributes[self.name]

    def __set_name__(self, cls, name):
        if name in cls.__annotations__:
            # If type is specified in annotation, take it from there.
            self.typ = cls.__annotations__[name]
            self.tadapter = pydantic.TypeAdapter(self.typ)

            # Update the recorded return type as well.
            # TODO: cannot do this at this point as the below dict is not yet populated
            # if name in cls.model_computed_fields:
            #     cls.model_computed_fields[name].return_type = self.typ

            # Have to remove it as pydantic will complain about overriding fields with computed fields.
            del cls.__annotations__[name]


class TypedSpan(Span):
    """A span with a type."""

    span_type: ClassVar[SpanType] = SpanType.UNKNOWN

    @staticmethod
    def attribute_property_factory(base: str) -> Callable:
        def prop_factory(
            name: str,
            typ: Optional[Type[T]] = None,
            typ_factory: Optional[Callable[[], Type[T]]] = None,
            default: Optional[T] = None,
            default_factory: Optional[Callable[[], T]] = None,
        ) -> property:
            return TypedSpan.attribute_property(
                name=base + "." + name,
                typ=typ,
                typ_factory=typ_factory,
                default=default,
                default_factory=default_factory,
            )

        return prop_factory

    @staticmethod
    def attribute_property(
        name: str,
        typ: Optional[Type[T]] = None,
        typ_factory: Optional[Callable[[], Type[T]]] = None,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None,
    ) -> property:
        """See AttributeProperty."""

        return pydantic.computed_field(
            AttributeProperty(name, typ, typ_factory, default, default_factory),
            return_type=typ,
        )


class EvalRoot(TypedSpan):
    """Root of feedback function evaluation."""

    span_type: ClassVar[SpanType] = SpanType.EVAL_ROOT

    # feedback result fields


class TraceRoot(TypedSpan):
    """Root of a trace."""

    span_type: ClassVar[SpanType] = SpanType.TRACE_ROOT

    # record fields


class Retrieval(TypedSpan):
    """A retrieval."""

    span_type: ClassVar[SpanType] = SpanType.RETRIEVAL

    query_text = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.QUERY_TEXT, str
    )
    """Input text whose related contexts are being retrieved."""

    query_embedding = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.QUERY_EMBEDDING, List[float]
    )
    """Embedding of the input text."""

    distance_type = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.DISTANCE_TYPE, str
    )
    """Distance function used for ranking contexts."""

    num_contexts = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.NUM_CONTEXTS, int
    )
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS, List[str]
    )
    """The retrieved contexts."""

    retrieved_scores = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_SCORES, List[float]
    )
    """The scores of the retrieved contexts."""

    retrieved_embeddings = TypedSpan.attribute_property(
        trace.SpanAttributes.RETRIEVAL.RETRIEVED_EMBEDDINGS, List[List[float]]
    )
    """The embeddings of the retrieved contexts."""


class Reranking(TypedSpan):
    """A reranker call."""

    span_type: ClassVar[SpanType] = SpanType.RERANKING

    query_text = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.QUERY_TEXT, str
    )
    """The query text."""

    model_name = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.MODEL_NAME, str
    )  # consider generic ML model name attr
    """The model name of the reranker."""

    top_n = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.TOP_N, int
    )
    """The number of contexts to rerank."""

    input_context_texts = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.INPUT_CONTEXT_TEXTS, List[str]
    )
    """The contexts being reranked."""

    input_context_scores = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.INPUT_CONTEXT_SCORES,
        Optional[List[float]],
    )
    """The scores of the input contexts."""

    output_ranks = TypedSpan.attribute_property(
        trace.SpanAttributes.RERANKING.OUTPUT_RANKS, List[int]
    )
    """Reranked indexes into `input_context_texts`."""


class Generation(TypedSpan):
    """A generation call to an LLM."""

    span_type: ClassVar[SpanType] = SpanType.GENERATION

    model_name = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.MODEL_NAME, str
    )  # to replace with otel's LLM_REQUEST_MODEL
    """The model name of the LLM."""

    model_type = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.MODEL_TYPE, str
    )
    """The type of model used."""

    input_token_count = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.INPUT_TOKEN_COUNT, int
    )  # to replace with otel's LLM_RESPONSE_USAGE_PROMPT_TOKENS
    """The number of tokens in the input."""

    input_messages = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.INPUT_MESSAGES, List[dict]
    )
    """The prompt given to the LLM."""

    output_token_count = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.OUTPUT_MESSAGES, int
    )  # to replace with otel's LLM_RESPONSE_COMPLETION_TOKENS
    """The number of tokens in the output."""

    output_messages = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.OUTPUT_MESSAGES, List[dict]
    )
    """The returned text."""

    temperature = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.TEMPERATURE, float
    )  # to replace with otel's LLM_REQUEST_TEMPERATURE
    """The temperature used for generation."""

    cost = TypedSpan.attribute_property(
        trace.SpanAttributes.GENERATION.COST, float
    )
    """The cost of the generation."""


class Memorization(TypedSpan):
    """A memory call."""

    span_type: ClassVar[SpanType] = SpanType.MEMORIZATION

    memory_type = TypedSpan.attribute_property(
        trace.SpanAttributes.MEMORIZATION.MEMORY_TYPE, str
    )
    """The type of memory."""

    remembered = TypedSpan.attribute_property(
        trace.SpanAttributes.MEMORIZATION.REMEMBERED, str
    )
    """The text being integrated into the memory in this span."""


class Embedding(TypedSpan):
    """An embedding call."""

    span_type: ClassVar[SpanType] = SpanType.EMBEDDING

    input_text = TypedSpan.attribute_property(
        trace.SpanAttributes.EMBEDDING.INPUT_TEXT, str
    )
    """The text being embedded."""

    model_name = TypedSpan.attribute_property(
        trace.SpanAttributes.EMBEDDING.MODEL_NAME, str
    )
    """The model name of the embedding model."""

    embedding = TypedSpan.attribute_property(
        trace.SpanAttributes.EMBEDDING.EMBEDDING, List[float]
    )
    """The embedding of the input text."""


class ToolInvocation(TypedSpan):
    """A tool invocation."""

    span_type: ClassVar[SpanType] = SpanType.TOOL_INVOCATION

    description = TypedSpan.attribute_property(
        trace.SpanAttributes.TOOL_INVOCATION.DESCRIPTION, str
    )
    """The description of the tool."""


class AgentInvocation(TypedSpan):
    """An agent invocation."""

    span_type: ClassVar[SpanType] = SpanType.AGENT_INVOCATION

    description = TypedSpan.attribute_property(
        trace.SpanAttributes.AGENT_INVOCATION.DESCRIPTION, str
    )
    """The description of the agent."""
