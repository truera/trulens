"""Spans

These are roughly equivalent to `RecordAppCall` but abstract away specific method
information into type of call related to types of components.
"""

from __future__ import annotations

import dataclasses
import datetime
from enum import Enum
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


from opentelemetry.semconv.ai import SpanAttributes as ai_span
from opentelemetry.util import types as ot_types
import pandas as pd
from pydantic import computed_field
from pydantic import Field
from pydantic import TypeAdapter

from trulens_eval import trace as mod_trace
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.utils import containers as mod_container_utils

logger = getLogger(__name__)

T = TypeVar("T")

class Span(mod_trace.OTSpan):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    @staticmethod
    def attribute_property(
        name: str,
        typ: Optional[Type[T]] = None,
        typ_factory: Optional[Callable[[], Type[T]]] = None,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None
    ) -> property:
        """Utility for creating properties that stores their values in the
        attributes dictionary with a vendor prefix.

        Validates default and on assignment.
        
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
        initialized = False
        tadapter = None

        def initialize():
            # Delaying the steps in this method until the first time the
            # property is used as otherwise forward references might not be
            # ready.

            nonlocal initialized, tadapter

            if initialized:
                return

            nonlocal typ, default
            if typ is None and typ_factory is not None:
                typ = typ_factory()

            if default is None and default_factory is not None:
                default = default_factory()

            if typ is None and default is not None:
                typ = type(default)

            if typ is None:
                tadapter = None
            else:
                tadapter = TypeAdapter(typ)
                if default is not None:
                    tadapter.validate_python(default)

        def getter(self) -> T:
            initialize()
            return self.attributes.get(self.vendor_attr(name), default)

        def setter(self, value: T) -> None:
            initialize()
            if tadapter is not None:
                tadapter.validate_python(value)

            self.attributes[self.vendor_attr(name)] = value

        prop = property(getter, setter)

        return computed_field(prop)

    @property
    def start_datetime(self) -> datetime.datetime:
        """Start time of span as a [datetime][datetime.datetime]."""
        return mod_container_utils.datetime_of_ns_timestamp(self.start_timestamp)
    
    @start_datetime.setter
    def start_datetime(self, value: datetime.datetime):
        self.start_timestamp = mod_container_utils.ns_timestamp_of_datetime(value)

    @property
    def end_datetime(self) -> datetime.datetime:
        """End time of span as a [datetime][datetime.datetime]."""
        return mod_container_utils.datetime_of_ns_timestamp(self.end_timestamp)
    
    @end_datetime.setter
    def end_datetime(self, value: datetime.datetime):
        self.end_timestamp = mod_container_utils.ns_timestamp_of_datetime(value)

    @property
    def span_id(self) -> mod_trace.TSpanID:
        """Identifier for the span."""

        return self.context.span_id

    @property
    def trace_id(self) -> mod_trace.TTraceID:
        """Identifier for the trace this span belongs to."""

        return self.context.trace_id

    @property # want # @functools.cached_property but those are not allowed to have setters
    def parent_context(self) -> Optional[mod_trace.HashableSpanContext]:
        """Context of parent span if any.

        This is stored in OT links with a relationship attribute of "parent".
        None if this is a root span or otherwise it does not have a parent.
        """

        for link_context, link_attributes in self.links.items():
            if link_attributes.get(self.vendor_attr("relationship")) == "parent":
                return link_context

        return None

    @parent_context.setter
    def parent_context(self, value: Optional[mod_trace.HashableSpanContext]):
        if value is None:
            return

        if self.parent_context is not None:
            # Delete existing parent if any.
            del self.links[self.parent_context]

        self.add_link(value, {self.vendor_attr("relationship"): "parent"})

    # want functools.cached_property but need updating due to the above setter
    @property
    def parent_span_id(self) -> Optional[mod_trace.TSpanID]:
        """Id of parent span if any."""

        parent_context = self.parent_context
        if parent_context is not None:
            return parent_context.span_id

        return None

    tags = attribute_property(
        "tags", typ=List[str], default_factory=list
    )
    """Tags associated with the span."""

    span_type = attribute_property(
        "span_type",
        typ_factory=lambda: SpanType,
        default_factory=lambda: SpanType.UNTYPED
    )
    """Type of span."""

    attributes_metadata: mod_container_utils.DictNamespace[ot_types.AttributeValue]
    # will be set as a DictNamespace indexing elements in attributes
    @property
    def metadata(self) -> mod_container_utils.DictNamespace[ot_types.AttributeValue]:
        return self.attributes_metadata

    @metadata.setter
    def metadata(self, value: Dict[str, str]):
        for k, v in value.items():
            self.attributes_metadata[k] = v

    def __init__(self, **kwargs):
        kwargs['attributes_metadata'] = mod_container_utils.DictNamespace(parent={}, namespace="temp")
        # Temporary fake for validation in super.__init__ below.

        super().__init__(**kwargs)

        # Actual. This is needed as pydantic will copy attributes dict in init.
        self.attributes_metadata = mod_container_utils.DictNamespace(
            parent=self.attributes,
            namespace=self.vendor_attr("metadata")
        )

        self.set_attribute(self.vendor_attr("span_type"), self.__class__.__name__)

class SpanUntyped(Span):
    """Generic span type.
    
    This represents spans that are being recorded but have not yet been
    determined to be of a particular type.
    """

class TransSpanRecord(Span):
    """A span whose activity was recorded in a record.
    
    Features references to the record.

    !!! note
        This is a transitional type for the traces work.
    """

    record: mod_record_schema.Record = Field(exclude=True, default=None)
    record_id = Span.attribute_property("record_id", typ=str, default=None)

class SpanMethodCall(TransSpanRecord):
    """Span which corresponds to a method call.
    
    See also temporary development attributes in
    [TransSpanRecordAppCall][trulens_eval.trace.span.TransSpanRecordCall].
    """

    inputs = Span.attribute_property("inputs", typ=Optional[Dict[str, Any]], default_factory=None)
    # TODO: Need to encode to OT AttributeValue

    output = Span.attribute_property("output", typ=Optional[Any], default_factory=None)
    # TODO: Need to encode to OT AttributeValue

    error = Span.attribute_property("error", typ=Optional[Any], default_factory=None)
    # TODO: Need to encode to OT AttributeValue


class TransSpanRecordAppCall(SpanMethodCall):
    """A Span which corresponds to single
    [RecordAppCall][trulens_eval.schema.record.RecordAppCall].

    Features references to the call.

    !!! note
        This is a transitional type for the traces work. The non-transitional
        fields are being placed in
        [SpanMethodCall][trulens_eval.trace.span.SpanMethodCall] instead.
    """
    call: mod_record_schema.RecordAppCall = Field(exclude=True, default=None)

class SpanRoot(TransSpanRecord):
    """A root span encompassing some collection of spans.

    Does not indicate any particular activity by itself beyond its children.
    """

SpanTyped = TransSpanRecordAppCall
"""Alias for the superclass of spans that went through the record call conversion."""

"""
@dataclasses.dataclass
class RetrieverQuery:
    text: str
    embedding: Optional[List[float]]

@dataclasses.dataclass
class RetrieverContext:
    text: str
    score: Optional[float]
    embedding: Optional[List[float]]
"""

class SpanRetriever(SpanTyped):
    """A retrieval."""

    query_text = Span.attribute_property("query_text", str)
    """Input text whose related contexts are being retrieved."""

    query_embedding = Span.attribute_property("query_embedding", List[float])
    """Embedding of the input text."""

    distance_type = Span.attribute_property("distance_type", str)
    """Distance function used for ranking contexts."""

    num_contexts = Span.attribute_property("num_contexts", int)
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = Span.attribute_property("retrieved_contexts", List[str])
    """The retrieved contexts."""

    retrieved_scores = Span.attribute_property("retrieved_scores", List[float])
    """The scores of the retrieved contexts."""

    retrieved_embeddings = Span.attribute_property("retrieved_embeddings", List[List[float]])
    """The embeddings of the retrieved contexts."""

class SpanVectorDBOTEL(SpanTyped):
    """VectorDB attributes from OpenTelemetry Semantic Conventions for AI.

    See [OpenTelemetry Semantic Conventions for AI
    constants](https://github.com/traceloop/openllmetry/blob/main/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv/ai/__init__.py)
    """

    # span attributes

    vector_db_vendor = Span.attribute_property(ai_span.VECTOR_DB_VENDOR, str) # optionality not known

    vector_db_query_top_k = Span.attribute_property(ai_span.VECTOR_DB_QUERY_TOP_K, int) # optionality not known

    # events

    """
    ai_span.DB_QUERY_EMBEDDINGS
    ai_span.DB_QUERY_RESULT
    """

    # event attributes

    """
    ai_span.DB_QUERY_EMBEDDINGS_VECTOR
    ai_span.DB_QUERY_RESULT_ID
    ai_span.DB_QUERY_RESULT_SCORE
    ai_span.DB_QUERY_RESULT_DISTANCE
    ai_span.DB_QUERY_RESULT_METADATA
    ai_span.DB_QUERY_RESULT_VECTOR
    ai_span.DB_QUERY_RESULT_DOCUMENT
    """

class SpanVectorDB(SpanVectorDBOTEL, SpanTyped):
    pass

class SpanReranker(SpanTyped):
    """A reranker call."""

    query_text = Span.attribute_property("query_text", str)
    """The query text."""

    model_name = Span.attribute_property("model_name", str)
    """The model name of the reranker."""

    top_n = Span.attribute_property("top_n", int)
    """The number of contexts to rerank."""

    input_context_texts = Span.attribute_property("input_context_texts", List[str])
    """The contexts being reranked."""

    input_context_scores = Span.attribute_property("input_context_scores", Optional[List[float]])
    """The scores of the input contexts."""

    output_ranks = Span.attribute_property("output_ranks", List[int])
    """Reranked indexes into `input_context_texts`."""

class SpanLLMOTEL(SpanTyped):
    """LLM attributes from OpenTelemetry Semantic Conventions for AI.

    See Open Telemetry Semantic Convetions for AI
    [llm_spans.md](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md)
    and
    [constants](https://github.com/traceloop/openllmetry/blob/main/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv/ai/__init__.py)

    There is also Arize's [openinference conventions](https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py).
    """

    request_model = Span.attribute_property(ai_span.LLM_REQUEST_MODEL, str)

    # system = Span.attribute_property(ai_span.LLM_REQUEST_SYSTEM, str)

    request_max_tokens = Span.attribute_property(ai_span.LLM_REQUEST_MAX_TOKENS, Optional[int])

    request_temperature = Span.attribute_property(ai_span.LLM_REQUEST_TEMPERATURE, Optional[float])

    request_top_p = Span.attribute_property(ai_span.LLM_REQUEST_TOP_P, Optional[float])

    response_finish_reasons = Span.attribute_property("gen_ai.response.finish_reasons", Optional[List[str]])
    # Not yet a constant in otel sem ai

    # response_id = Span.attribute_property("gen_ai.response.id", Optional[str])
    # Not yet a constant in otel sem ai

    response_model = Span.attribute_property(ai_span.LLM_RESPONSE_MODEL, Optional[str])

    # usage_completion_tokens = Span.attribute_property(ai_span.LLM_RESPONSE_COMPLETION_TOKENS, Optional[int])

    # usage_promot_tokens = Span.attribute_property(ai_span.LLM_RESPONSE_USAGE_PROMPT_TOKENS, Optional[int])

    """
    # These below are in the constants file but are not described in the
    # markdown so their details or optionality is not know.

    request_type = Span.attribute_property(ai_span.LLM_REQUEST_TYPE, Optional[str]) # optionality not known

    usage_total_tokens = Span.attribute_property(ai_span.LLM_USAGE_TOTAL_TOKENS, Optional[int]) # optionality not known

    user = Span.attribute_property(ai_span.LLM_REQUEST_USER, Optional[str]) # optionality not known

    headers = Span.attribute_property(ai_span.LLM_REQUEST_HEADERS, Optional[str]) # optionality, type not known

    top_k = Span.attribute_property(ai_span.LLM_REQUEST_TOP_K, Optional[int]) # optionality not known

    is_streaming = Span.attribute_property(ai_span.LLM_REQUEST_IS_STREAMING, Optional[bool]) # optionality not known

    frequency_penalty = Span.attribute_property(ai_span.LLM_REQUEST_FREQUENCY_PENALTY, Optional[float]) # optionality not known

    presence_penalty = Span.attribute_property(ai_span.LLM_REQUEST_PRESENCE_PENALTY, Optional[float]) # optionality not known

    chat_stop_sequences = Span.attribute_property(ai_span.LLM_REQUEST_CHAT_STOP_SEQUENCES, Optional[List[str]]) # optionality, type not known

    request_functions = Span.attribute_property(ai_span.LLM_REQUEST_FUNCTIONS, Optional[List[str]]) # optionality, type not known
    """

    # Events in description but not yet in python package
    # Events named "gen_ai.content.prompt"
    # Attributes:
    # - "gen_ai.prompt": str (json data)
    # - "gen_ai.completion": str (json data)

class SpanLLM(SpanLLMOTEL, SpanTyped):
    """A generation call to an LLM.
    
    This features attributes not covered by the OpenTelemetry Semantic
    Conventions for AI attributes in [SpanLLMOtel][trulens_eval.trace.span.SpanLLMOtel].
    """

    model_type = Span.attribute_property("model_type", str)
    """The type of model used."""

    input_messages = Span.attribute_property("input_messages", List[dict])
    """The prompt given to the LLM."""

    output_messages = Span.attribute_property("output_messages", List[dict])
    """The returned text."""

    cost = Span.attribute_property("cost", float)
    """The cost of the generation."""

class SpanMemory(SpanTyped):
    """A memory call."""

    memory_type = Span.attribute_property("memory_type", str)
    """The type of memory."""

    remembered = Span.attribute_property("remembered", str)
    """The text being integrated into the memory in this span."""

class SpanEmbedding(SpanTyped):
    """An embedding cal."""

    input_text = Span.attribute_property("input_text", str)
    """The text being embedded."""

    model_name = Span.attribute_property("model_name", str)
    """The model name of the embedding model."""

    embedding = Span.attribute_property("embedding", List[float])
    """The embedding of the input text."""

class SpanTool(SpanTyped):
    """A tool invocation."""

    description = Span.attribute_property("description", str)
    """The description of the tool."""

class SpanAgent(SpanTyped):
    """An agent invocation."""

    description = Span.attribute_property("description", str)
    """The description of the agent."""

class SpanTask(SpanTyped):
    """A task invocation."""

class SpanOther(SpanTyped):
    """Other uncategorized spans."""

class SpanType(Enum):
    """Span types.
    
    This is a bit redundant with the span type hierarchy above. It is here for
    convenience of looking up types in means other than `__class__` or via
    `isinstance`.
    """

    def to_class(self) -> Type[Span]:
        """Convert to the class for this type."""

        if hasattr(mod_trace.span, self.value):
            return getattr(mod_trace.span, self.value)
        
        raise ValueError(f"Span type {self.value} not found in module.")

    UNTYPED = SpanUntyped.__name__
    """See [SpanUntyped][trulens_eval.trace.span.SpanUntyped]."""

    ROOT = SpanRoot.__name__
    """See [SpanRoot][trulens_eval.trace.span.SpanRoot]."""

    RETRIEVER = SpanRetriever.__name__
    """See [SpanRetriever][trulens_eval.trace.span.SpanRetriever]."""

    RERANKER = SpanReranker.__name__
    """See [SpanReranker][trulens_eval.trace.span.SpanReranker]."""

    LLM = SpanLLM.__name__
    """See [SpanLLM][trulens_eval.trace.span.SpanLLM]."""

    MEMORY = SpanMemory.__name__
    """See [SpanMemory][trulens_eval.trace.span.SpanMemory]."""

    EMBEDDING = SpanEmbedding.__name__
    """See [SpanEmbedding][trulens_eval.trace.span.SpanEmbedding]."""

    TOOL = SpanTool.__name__
    """See [SpanTool][trulens_eval.trace.span.SpanTool]."""

    AGENT = SpanAgent.__name__
    """See [SpanAgent][trulens_eval.trace.span.SpanAgent]."""

    TASK = SpanTask.__name__
    """See [SpanTask][trulens_eval.trace.span.SpanTask]."""

    OTHER = SpanOther.__name__
    """See [SpanOther][trulens_eval.trace.span.SpanOther]."""
