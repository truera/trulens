from __future__ import annotations

import datetime
import functools
import inspect
from logging import getLogger
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)
import uuid

import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing.core import trace as core_trace
from trulens.semconv import trace as truconv

logger = getLogger(__name__)

T = TypeVar("T")


class AttributeProperty(property, Generic[T]):
    """Property that stores a serialized version its value in the attributes
    dictionary.

    Validates default and on assignment. This is meant to be used only in
    TypedSpan instances (or subclasses).

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
            try:
                self.tadapter = pydantic.TypeAdapter(self.typ)

                if self.default is not None:
                    self.tadapter.validate_python(self.default)

            except pydantic.PydanticSchemaGenerationError:
                self.tadapter = None

    def fget(self, obj: Any) -> Optional[T]:
        return self.__get__(obj, obj.__class__)

    def __get__(self, obj: Any, objtype: Optional[Type[T]]) -> Optional[T]:  # type: ignore # noqa: F821
        if obj is None:
            return self

        self.init_forward()
        return obj._attributes.get(self.name, self.default)

    def __set__(self, obj, value: T) -> None:
        self.init_forward()

        if self.tadapter is not None:
            self.tadapter.validate_python(value)

        obj._attributes[self.name] = value
        obj.attributes[self.name] = json_utils.jsonify(value)

    def __delete__(self, obj):
        del obj._attributes[self.name]
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


@functools.lru_cache
def _get_combo_class(classes: Tuple[Type[TypedSpan]]) -> Type[TypedSpan]:
    """Get the class that corresponds to the combination of classes in the
    input set.

    Args:
        classes: The set of classes to combine.

    Returns:
        The class that corresponds to the combination of the input classes.
    """

    class _Combo(*classes):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.span_types = set(s.span_type for s in classes)

    _Combo.__name__ = "_".join(cls.__name__ for cls in classes)
    _Combo.__qualname__ = "_".join(cls.__qualname__ for cls in classes)

    return _Combo


def get_combo_class(classes: Set[Type[TypedSpan]]) -> Type[TypedSpan]:
    """Get the class that corresponds to the combination of classes in the
    input set.

    Also populates the span_types field.

    Args:
        classes: The set of classes to combine.

    Returns:
        The class that corresponds to the combination of the input classes.
    """

    classes = tuple(classes)

    if len(classes) == 1:
        return classes[0]

    classes = tuple(sorted(classes, key=lambda cls: cls.__qualname__))

    return _get_combo_class(classes)


class TypedSpan(core_trace.Span):
    """A span with a type."""

    span_type: ClassVar[Optional[truconv.SpanAttributes.SpanType]] = None
    """Mixin type for each subclass."""

    _attributes: Dict[str, Any] = pydantic.PrivateAttr(default_factory=dict)
    """Non-serialized values of named fields defined by `attribute_property`.

    These are mirrored with serialized versions in `attributes`.
    """

    @classmethod
    def mixin_new(cls, d: Optional[Dict] = None, **kwargs) -> TypedSpan:
        """Given a jsonized version of a typed span that may be of multiple
        types, initialize the appropriate classes with the provided data."""

        if d is None:
            d = {}

        d.update(kwargs)

        types = d.pop("span_types", [])

        classes = {TYPE_TO_CLASS_MAP[t] for t in types}

        combo_class = get_combo_class(classes)

        return combo_class(**d)

    @staticmethod
    def semanticize(span: core_trace.Span) -> TypedSpan:
        class_args = {
            "name": span.name,
            "start_timestamp": span.start_timestamp,
            "end_timestamp": span.end_timestamp,
            "attributes": span.attributes,
            "status": span.status,
            "status_description": span.status_description,
            "links": span.links,
            "events": span.events,
        }

        classes = set()

        if isinstance(span, core_trace.LiveSpanCall):
            classes.add(Call)
            classes.add(Semantic)

            class_args.update(
                dict(
                    call_id=span.call_id,
                    signature=pyschema_utils.Signature.of_signature(
                        span.live_sig
                    ),
                    function=pyschema_utils.FunctionOrMethod.of_callable(
                        span.live_func
                    ),
                    process_id=span.process_id,
                    thread_id=span.thread_id,
                    bindings=pyschema_utils.Bindings.of_bound_arguments(
                        span.live_bindings
                    ),
                    ret=json_utils.jsonify(span.live_ret),
                )
            )

        if isinstance(span, core_trace.WithCost):
            classes.add(Cost)
            classes.add(Semantic)

            class_args["cost"] = span.cost

        if isinstance(span, core_trace.LiveAppCall):
            classes.add(App)
            classes.add(Semantic)

            class_args["app_ids"] = set(app.app_id for app in span.live_apps)
            class_args["record_ids"] = span.record_ids

        if isinstance(span, core_trace.LiveRecordRoot):
            classes.add(RecordRoot)

            class_args.update(dict(record_id=span.trace_record_id))

            app = span.live_app()
            if app is None:
                logger.warning(
                    "App in %s got garbage collected before serialization.",
                    span,
                )
            else:
                class_args.update(
                    dict(
                        app_id=app.app_id,
                        app_name=app.app_name,
                        app_version=app.app_version,
                    )
                )

        if len(classes) == 0:
            logger.warning("No types found for span %s.", span)
            classes.add(Unknown)

        cls = get_combo_class(classes)

        instance = cls(**class_args)

        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This is meant to be overwritten:
        self.span_types = set([self.__class__.span_type])

        for name, value in kwargs.items():
            if name in self.__class__.model_computed_fields:
                setattr(self, name, value)

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

    span_types = attribute_property(
        truconv.SpanAttributes.SPAN_TYPES,
        Set[truconv.SpanAttributes.SpanType],
        default_factory=set,
    )
    """A span can be of multiple semantic categories."""


class Unknown(TypedSpan):
    """An unknown span type."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.UNKNOWN
    )


class EvalRoot(TypedSpan):
    """Root of feedback function evaluation."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.EVAL_ROOT
    )

    # feedback result fields


class RecordRoot(TypedSpan):
    """Root of a record."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RECORD_ROOT
    )

    # TODEP:
    record_id = TypedSpan.attribute_property(
        "record.record_id", types_schema.RecordID
    )
    # TODEP:
    perf = TypedSpan.attribute_property("record.perf", base_schema.Perf)
    # TODEP:
    ts = TypedSpan.attribute_property("record.ts", datetime.datetime)

    app_id = TypedSpan.attribute_property("record.app_id", types_schema.AppID)
    app_name = TypedSpan.attribute_property("record.app_name", str)
    app_version = TypedSpan.attribute_property("record.app_version", str)

    cost = TypedSpan.attribute_property("record.cost", base_schema.Cost)

    tags = TypedSpan.attribute_property("record.tags", str)

    meta = TypedSpan.attribute_property("record.meta", serial_utils.JSON)

    main_input = TypedSpan.attribute_property(
        "record.main_input", serial_utils.JSON
    )

    main_output = TypedSpan.attribute_property(
        "record.main_output", serial_utils.JSON
    )

    main_error = TypedSpan.attribute_property(
        "record.main_error", serial_utils.JSON
    )


class Semantic(TypedSpan):
    """A normal span that is not unknown."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.SEMANTIC
    )


class Cost(TypedSpan):
    """A span that corresponds to a cost."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.COST
    )

    cost = TypedSpan.attribute_property(
        truconv.SpanAttributes.COST.COST, base_schema.Cost
    )


class Call(TypedSpan):
    """A typed span that corresponds to a method call."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.CALL
    )

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    call_id = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.CALL_ID, uuid.UUID
    )
    """Unique identifier for the call."""

    signature = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.SIGNATURE, inspect.Signature
    )
    """Signature of the function."""

    function = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.FUNCTION, pyschema_utils.FunctionOrMethod
    )
    """Function info."""

    # TODO: move this to resource attributes:
    process_id = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.PROCESS_ID, int
    )
    """Process id."""

    thread_id = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.THREAD_ID, int
    )
    """Thread id."""

    bindings = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.BINDINGS, pyschema_utils.Bindings
    )
    """Bindings of the function, if can be bound."""

    ret = TypedSpan.attribute_property(
        truconv.SpanAttributes.CALL.RETURN, serial_utils.JSON
    )


class App(TypedSpan):
    """A typed span that corresponds to an app call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.APP
    )

    app_ids = TypedSpan.attribute_property(
        truconv.SpanAttributes.APP.APP_IDS, Set[types_schema.AppID]
    )
    """The app ids of the apps that were called."""

    record_ids = TypedSpan.attribute_property(
        truconv.SpanAttributes.APP.RECORD_IDS,
        Dict[types_schema.AppID, types_schema.RecordID],
    )
    """The map of app_id to record_id indicating the id of the span as viewed by
    each app that was tracing it."""


class Retrieval(TypedSpan):
    """A retrieval."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RETRIEVAL
    )

    query_text = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.QUERY_TEXT, str
    )
    """Input text whose related contexts are being retrieved."""

    query_embedding = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.QUERY_EMBEDDING, List[float]
    )
    """Embedding of the input text."""

    distance_type = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.DISTANCE_TYPE, str
    )
    """Distance function used for ranking contexts."""

    num_contexts = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.NUM_CONTEXTS, int
    )
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS, List[str]
    )
    """The retrieved contexts."""

    retrieved_scores = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_SCORES, List[float]
    )
    """The scores of the retrieved contexts."""

    retrieved_embeddings = TypedSpan.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_EMBEDDINGS, List[List[float]]
    )
    """The embeddings of the retrieved contexts."""


class Reranking(TypedSpan):
    """A reranker call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RERANKING
    )

    query_text = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.QUERY_TEXT, str
    )
    """The query text."""

    model_name = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.MODEL_NAME, str
    )  # consider generic ML model name attr
    """The model name of the reranker."""

    top_n = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.TOP_N, int
    )
    """The number of contexts to rerank."""

    input_context_texts = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.INPUT_CONTEXT_TEXTS, List[str]
    )
    """The contexts being reranked."""

    input_context_scores = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.INPUT_CONTEXT_SCORES,
        Optional[List[float]],
    )
    """The scores of the input contexts."""

    output_ranks = TypedSpan.attribute_property(
        truconv.SpanAttributes.RERANKING.OUTPUT_RANKS, List[int]
    )
    """Reranked indexes into `input_context_texts`."""


class Generation(TypedSpan):
    """A generation call to an LLM."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.GENERATION
    )

    model_name = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.MODEL_NAME, str
    )  # to replace with otel's LLM_REQUEST_MODEL
    """The model name of the LLM."""

    model_type = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.MODEL_TYPE, str
    )
    """The type of model used."""

    input_token_count = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.INPUT_TOKEN_COUNT, int
    )  # to replace with otel's LLM_RESPONSE_USAGE_PROMPT_TOKENS
    """The number of tokens in the input."""

    input_messages = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.INPUT_MESSAGES, List[dict]
    )
    """The prompt given to the LLM."""

    output_token_count = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.OUTPUT_MESSAGES, int
    )  # to replace with otel's LLM_RESPONSE_COMPLETION_TOKENS
    """The number of tokens in the output."""

    output_messages = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.OUTPUT_MESSAGES, List[dict]
    )
    """The returned text."""

    temperature = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.TEMPERATURE, float
    )  # to replace with otel's LLM_REQUEST_TEMPERATURE
    """The temperature used for generation."""

    cost = TypedSpan.attribute_property(
        truconv.SpanAttributes.GENERATION.COST, float
    )
    """The cost of the generation."""


class Memorization(TypedSpan):
    """A memory call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.MEMORIZATION
    )

    memory_type = TypedSpan.attribute_property(
        truconv.SpanAttributes.MEMORIZATION.MEMORY_TYPE, str
    )
    """The type of memory."""

    remembered = TypedSpan.attribute_property(
        truconv.SpanAttributes.MEMORIZATION.REMEMBERED, str
    )
    """The text being integrated into the memory in this span."""


class Embedding(TypedSpan):
    """An embedding call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.EMBEDDING
    )

    input_text = TypedSpan.attribute_property(
        truconv.SpanAttributes.EMBEDDING.INPUT_TEXT, str
    )
    """The text being embedded."""

    model_name = TypedSpan.attribute_property(
        truconv.SpanAttributes.EMBEDDING.MODEL_NAME, str
    )
    """The model name of the embedding model."""

    embedding = TypedSpan.attribute_property(
        truconv.SpanAttributes.EMBEDDING.EMBEDDING, List[float]
    )
    """The embedding of the input text."""


class ToolInvocation(TypedSpan):
    """A tool invocation."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.TOOL_INVOCATION
    )

    description = TypedSpan.attribute_property(
        truconv.SpanAttributes.TOOL_INVOCATION.DESCRIPTION, str
    )
    """The description of the tool."""


class AgentInvocation(TypedSpan):
    """An agent invocation."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.AGENT_INVOCATION
    )

    description = TypedSpan.attribute_property(
        truconv.SpanAttributes.AGENT_INVOCATION.DESCRIPTION, str
    )
    """The description of the agent."""


TYPE_TO_CLASS_MAP: Dict[truconv.SpanAttributes.SpanType, Type[TypedSpan]] = {
    truconv.SpanAttributes.SpanType.UNKNOWN: Unknown,
    truconv.SpanAttributes.SpanType.SEMANTIC: Semantic,
    truconv.SpanAttributes.SpanType.EVAL_ROOT: EvalRoot,
    truconv.SpanAttributes.SpanType.RECORD_ROOT: RecordRoot,
    truconv.SpanAttributes.SpanType.APP: App,
    truconv.SpanAttributes.SpanType.CALL: Call,
    truconv.SpanAttributes.SpanType.COST: Cost,
    truconv.SpanAttributes.SpanType.RETRIEVAL: Retrieval,
    truconv.SpanAttributes.SpanType.RERANKING: Reranking,
    truconv.SpanAttributes.SpanType.GENERATION: Generation,
    truconv.SpanAttributes.SpanType.MEMORIZATION: Memorization,
    truconv.SpanAttributes.SpanType.EMBEDDING: Embedding,
    truconv.SpanAttributes.SpanType.TOOL_INVOCATION: ToolInvocation,
    truconv.SpanAttributes.SpanType.AGENT_INVOCATION: AgentInvocation,
}
"""Map of classes from their type enum."""
