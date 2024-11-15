"""TypedSpan organization and semantization.

The Span subclass and subsubclasses defined here are the only ones we put into the database.
"""

from __future__ import annotations

import functools
import inspect
from logging import getLogger
from typing import (
    ClassVar,
    Dict,
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
from trulens.experimental.otel_tracing.core import span as core_span
from trulens.semconv import trace as truconv

logger = getLogger(__name__)

T = TypeVar("T")


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

            # for cls in classes:
            #    cls._init_attribute_properties(self, kwargs)

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


class TypedSpan(core_span.Span):
    """A span with a type."""

    span_type: ClassVar[Optional[truconv.SpanAttributes.SpanType]] = None
    """Mixin type for each subclass."""

    @classmethod
    def mixin_new(cls, d: Optional[Dict] = None, **kwargs) -> TypedSpan:
        """Given a jsonized version of a typed span that may be of multiple
        types, initialize the appropriate classes with the provided data."""

        if d is None:
            d = {}

        d.update(kwargs)

        # if (record_ids := d.get("record_ids", None)) is not None:
        # Some span types have record_id field which gets stored in
        # record_ids in the db.
        #    if len(record_ids) == 1:
        #        d["record_id"] = next(iter(record_ids.values()))

        types = d.pop("span_types", [])

        classes = {TYPE_TO_CLASS_MAP[t] for t in types}

        combo_class = get_combo_class(classes)

        return combo_class(**d)

    @staticmethod
    def semanticize(span: core_span.Span) -> TypedSpan:
        class_args = {
            "context": span.context,
            "parent": span.parent,
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

        if isinstance(span, core_span.Span):
            classes.add(
                Record
            )  # everything that comes from trulens tracer is a span under an app right now
            class_args["record_ids"] = span.record_ids
            class_args["app_ids"] = span.app_ids

            if span.record_ids is None:
                raise RuntimeError("Span has no record_ids.")

        if isinstance(span, core_span.RecordingContextSpan):
            classes.add(Recording)

            app = span.live_app()
            if app is None:
                logger.warning(
                    "App in %s got garbage collected before serialization.",
                    span,
                )
            else:
                class_args.update(dict(app_id=app.app_id))

        if isinstance(span, core_span.LiveSpan):
            # already covered everything in the Span case. The other live objects are not relevant.
            pass

        if isinstance(span, core_span.LiveSpanCall):
            classes.add(Call)

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
                    bound_arguments=pyschema_utils.BoundArguments.of_bound_arguments(
                        span.live_bound_arguments
                    )
                    if span.live_bound_arguments is not None
                    else None,
                    ret=json_utils.jsonify(span.live_ret),
                    call_error=json_utils.jsonify(span.live_error),
                )
            )

        if isinstance(span, core_span.WithCost):
            classes.add(Cost)

            class_args["cost"] = span.cost

        if isinstance(span, core_span.LiveRecordRoot):
            classes.add(RecordRoot)

            class_args.update(dict(record_id=span.record_id))

            app = span.live_app()

            if app is None:
                logger.warning(
                    "App in %s got garbage collected before serialization.",
                    span,
                )
            else:
                # Get the main method call so we can get the main input/output/error.

                main_span = span.first_child(
                    matching=lambda s: isinstance(s, core_span.LiveSpanCall)
                )

                total_cost = span.cost_tally()

                if main_span is None:
                    logger.warning(
                        "No main span found for record %s in %s.",
                        span.record_id,
                        span,
                    )

                else:
                    main_input = app.main_input(
                        func=main_span.live_func,
                        sig=main_span.live_sig,
                        bindings=main_span.live_bound_arguments,
                    )
                    main_output = app.main_output(
                        func=main_span.live_func,
                        sig=main_span.live_sig,
                        bindings=main_span.live_bound_arguments,
                        ret=main_span.live_ret,
                    )
                    main_error = json_utils.jsonify(main_span.live_error)

                    class_args.update(
                        dict(
                            main_input=main_input,
                            main_output=main_output,
                            main_error=main_error,
                        )
                    )

                class_args.update(
                    dict(
                        app_id=app.app_id,
                        app_name=app.app_name,
                        app_version=app.app_version,
                        total_cost=total_cost,
                    )
                )

        # TODO:
        # classes.add(Semantic)

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

    span_types = core_span.Span.attribute_property(
        truconv.SpanAttributes.SPAN_TYPES,
        Set[truconv.SpanAttributes.SpanType],
        default_factory=set,
    )
    """A span can be of multiple categories."""


class Recording(TypedSpan):
    """A TruLens recording context span type."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RECORDING
    )

    app_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORDING.APP_ID, types_schema.AppID
    )


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

    record_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.RECORD_ID, types_schema.RecordID
    )

    # TODEP:
    # perf = core_span.Span.attribute_property("record.perf", base_schema.Perf)
    # TODEP:
    # ts = core_span.Span.attribute_property("record.ts", datetime.datetime)
    # tags = core_span.Span.attribute_property("record.tags", str)
    # meta = core_span.Span.attribute_property("record.meta", serial_utils.JSON)

    app_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.APP_ID, types_schema.AppID
    )
    app_name = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.APP_NAME, str
    )
    app_version = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.APP_VERSION, str
    )

    total_cost = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.TOTAL_COST, base_schema.Cost
    )

    main_input = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.MAIN_INPUT, serial_utils.JSON
    )

    main_output = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.MAIN_OUTPUT, serial_utils.JSON
    )

    main_error = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.MAIN_ERROR, serial_utils.JSON
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

    cost = core_span.Span.attribute_property(
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

    call_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.CALL_ID, uuid.UUID
    )
    """Unique identifier for the call."""

    signature = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.SIGNATURE, inspect.Signature
    )
    """Signature of the function."""

    function = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.FUNCTION, pyschema_utils.FunctionOrMethod
    )
    """Function info."""

    # TODO: move this to resource attributes:
    process_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.PROCESS_ID, int
    )
    """Process id."""

    thread_id = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.THREAD_ID, int
    )
    """Thread id."""

    bound_arguments = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.BOUND_ARGUMENTS,
        Optional[pyschema_utils.BoundArguments],
        default=None,
    )
    """Bindings of the function, if can be bound."""

    ret = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.RETURN, serial_utils.JSON
    )

    call_error = core_span.Span.attribute_property(
        truconv.SpanAttributes.CALL.ERROR, serial_utils.JSON
    )


class Record(TypedSpan):
    """Span that contains recording/app ids."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RECORD
    )

    app_ids = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD.APP_IDS, Set[types_schema.AppID]
    )
    """The app ids of the apps that were called."""

    record_ids = core_span.Span.attribute_property(
        truconv.SpanAttributes.RECORD.RECORD_IDS,
        Dict[types_schema.AppID, types_schema.RecordID],
    )
    """The map of app_id to record_id indicating the id of the span as viewed by
    each app that was tracing it."""


class Retrieval(TypedSpan):
    """A retrieval."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RETRIEVAL
    )

    query_text = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.QUERY_TEXT, str
    )
    """Input text whose related contexts are being retrieved."""

    query_embedding = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.QUERY_EMBEDDING, List[float]
    )
    """Embedding of the input text."""

    distance_type = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.DISTANCE_TYPE, str
    )
    """Distance function used for ranking contexts."""

    num_contexts = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.NUM_CONTEXTS, int
    )
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS, List[str]
    )
    """The retrieved contexts."""

    retrieved_scores = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_SCORES, List[float]
    )
    """The scores of the retrieved contexts."""

    retrieved_embeddings = core_span.Span.attribute_property(
        truconv.SpanAttributes.RETRIEVAL.RETRIEVED_EMBEDDINGS, List[List[float]]
    )
    """The embeddings of the retrieved contexts."""


class Reranking(TypedSpan):
    """A reranker call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.RERANKING
    )

    query_text = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.QUERY_TEXT, str
    )
    """The query text."""

    model_name = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.MODEL_NAME, str
    )  # consider generic ML model name attr
    """The model name of the reranker."""

    top_n = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.TOP_N, int
    )
    """The number of contexts to rerank."""

    input_context_texts = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.INPUT_CONTEXT_TEXTS, List[str]
    )
    """The contexts being reranked."""

    input_context_scores = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.INPUT_CONTEXT_SCORES,
        Optional[List[float]],
    )
    """The scores of the input contexts."""

    output_ranks = core_span.Span.attribute_property(
        truconv.SpanAttributes.RERANKING.OUTPUT_RANKS, List[int]
    )
    """Reranked indexes into `input_context_texts`."""


class Generation(TypedSpan):
    """A generation call to an LLM."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.GENERATION
    )

    model_name = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.MODEL_NAME, str
    )  # to replace with otel's LLM_REQUEST_MODEL
    """The model name of the LLM."""

    model_type = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.MODEL_TYPE, str
    )
    """The type of model used."""

    input_token_count = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.INPUT_TOKEN_COUNT, int
    )  # to replace with otel's LLM_RESPONSE_USAGE_PROMPT_TOKENS
    """The number of tokens in the input."""

    input_messages = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.INPUT_MESSAGES, List[dict]
    )
    """The prompt given to the LLM."""

    output_token_count = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.OUTPUT_MESSAGES, int
    )  # to replace with otel's LLM_RESPONSE_COMPLETION_TOKENS
    """The number of tokens in the output."""

    output_messages = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.OUTPUT_MESSAGES, List[dict]
    )
    """The returned text."""

    temperature = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.TEMPERATURE, float
    )  # to replace with otel's LLM_REQUEST_TEMPERATURE
    """The temperature used for generation."""

    cost = core_span.Span.attribute_property(
        truconv.SpanAttributes.GENERATION.COST, float
    )
    """The cost of the generation."""


class Memorization(TypedSpan):
    """A memory call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.MEMORIZATION
    )

    memory_type = core_span.Span.attribute_property(
        truconv.SpanAttributes.MEMORIZATION.MEMORY_TYPE, str
    )
    """The type of memory."""

    remembered = core_span.Span.attribute_property(
        truconv.SpanAttributes.MEMORIZATION.REMEMBERED, str
    )
    """The text being integrated into the memory in this span."""


class Embedding(TypedSpan):
    """An embedding call."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.EMBEDDING
    )

    input_text = core_span.Span.attribute_property(
        truconv.SpanAttributes.EMBEDDING.INPUT_TEXT, str
    )
    """The text being embedded."""

    model_name = core_span.Span.attribute_property(
        truconv.SpanAttributes.EMBEDDING.MODEL_NAME, str
    )
    """The model name of the embedding model."""

    embedding = core_span.Span.attribute_property(
        truconv.SpanAttributes.EMBEDDING.EMBEDDING, List[float]
    )
    """The embedding of the input text."""


class ToolInvocation(TypedSpan):
    """A tool invocation."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.TOOL_INVOCATION
    )

    description = core_span.Span.attribute_property(
        truconv.SpanAttributes.TOOL_INVOCATION.DESCRIPTION, str
    )
    """The description of the tool."""


class AgentInvocation(TypedSpan):
    """An agent invocation."""

    span_type: ClassVar[truconv.SpanAttributes.SpanType] = (
        truconv.SpanAttributes.SpanType.AGENT_INVOCATION
    )

    description = core_span.Span.attribute_property(
        truconv.SpanAttributes.AGENT_INVOCATION.DESCRIPTION, str
    )
    """The description of the agent."""


TYPE_TO_CLASS_MAP: Dict[truconv.SpanAttributes.SpanType, Type[TypedSpan]] = {
    truconv.SpanAttributes.SpanType.UNKNOWN: Unknown,
    truconv.SpanAttributes.SpanType.SEMANTIC: Semantic,
    truconv.SpanAttributes.SpanType.RECORDING: Recording,
    truconv.SpanAttributes.SpanType.EVAL_ROOT: EvalRoot,
    truconv.SpanAttributes.SpanType.RECORD_ROOT: RecordRoot,
    truconv.SpanAttributes.SpanType.RECORD: Record,
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
