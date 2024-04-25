"""Span types.

These are roughly equivalent to RecordAppCall but abstract away specific method
information into type of call related to types of components.
"""

from __future__ import annotations

import contextvars
import datetime
from enum import Enum
import functools
from logging import getLogger
from pprint import pprint
import random
import time
from typing import (Any, ClassVar, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, TypeVar, Union)

import opentelemetry
from opentelemetry.trace import status as trace_status
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
from opentelemetry.util._decorator import _agnosticcontextmanager
import pydantic
from pydantic import PlainSerializer
from pydantic.functional_validators import PlainValidator
from typing_extensions import Annotated

logger = getLogger(__name__)

# import trulens_eval

# Type alises

A = TypeVar("A")
B = TypeVar("B")

TTimestamp = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NumTimestampBits = 64

TSpanID = int # 
"""Type of span identifiers.

64 bit int as per OpenTelemetry.
"""
NumSpanIDBits = 64
"""Number of bits in a span identifier."""

TTraceID = int
"""Type of trace identifiers.

128 bit int as per OpenTelemetry.
"""
NumTraceIDBits = 128
"""Number of bits in a trace identifier."""

T = TypeVar("T")

class HashableSpanContext(ot_span.SpanContext):
    """SpanContext that can be hashed.

    Does not change data layout or behaviour. Changing SpanContext
    `__class__` with this should be safe.
    """

    def __hash__(self):
        return hash((self.trace_id, self.span_id))

    def __eq__(self, other):
        return self.trace_id == other.trace_id and self.span_id == other.span_id

def deserialize_contextmapping(v: List[Tuple[HashableSpanContext, T]]) -> Dict[HashableSpanContext, T]:
    """Deserialize a list of tuples as a dictionary."""

    return {HashableSpanContext(*k[0:-1]): v for k, v in v}
    
def serialize_contextmapping(
    v: Dict[HashableSpanContext, T],
) -> List[Tuple[A, B]]:
    """Serialize a dictionary as a list of tuples."""

    return list(v.items())

ContextMapping = Annotated[
    Dict[HashableSpanContext, T],
    PlainSerializer(serialize_contextmapping),
    PlainValidator(deserialize_contextmapping)
]
"""Type annotation for pydantic fields that store dictionaries whose keys are
HashableSpanContext.

This is needed to help pydantic figure out how to serialize and deserialize these dicts.
"""


def make_hashable(context: ot_span.SpanContext) -> HashableSpanContext:
    # HACK015: replace class of contexts to add hashing

    if context.__class__ is not HashableSpanContext:
        context.__class__ = HashableSpanContext

    # Return not needed but useful for type checker.
    return context

class OTSpan(pydantic.BaseModel, ot_span.Span):
    """Implementation of OpenTelemetry Span requirements.
    
    See also [OpenTelemetry Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span).
    """

    _vendor: ClassVar[str] = "trulens_eval"
    @classmethod
    def _attr(self, name):
        return f"{self._vendor}@{name}"

    model_config = {
        'arbitrary_types_allowed': True,
        'use_attribute_docstrings': True
    }
    """Pydantic configuration."""

    name: str
    """Name of span."""

    kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL
    """Kind of span."""

    status: trace_status.StatusCode = trace_status.StatusCode.UNSET
    """Status of the span as per OpenTelemetry Span requirements."""

    status_description: Optional[str] = None
    """Status description as per OpenTelemetry Span requirements."""

    start_timestamp: TTimestamp = pydantic.Field(default_factory=time.time_ns)
    """Timestamp when the span's activity started in nanoseconds since epoch."""

    end_timestamp: Optional[TTimestamp] = None
    """Timestamp when the span's activity ended in nanoseconds since epoch.

    None if not yet ended.
    """

    context: HashableSpanContext
    """Unique immutable identifier for the span."""


    events: List[Tuple[str, ot_types.Attributes, TTimestamp]] = pydantic.Field(default_factory=list)
    """Events recorded in the span."""

    links: ContextMapping[Mapping[str, ot_types.AttributeValue]] = pydantic.Field(default_factory=dict)
    """Relationships to other spans with attributes on each link."""

    attributes: Dict[str, ot_types.AttributeValue] = pydantic.Field(default_factory=dict)
    """Attributes of span."""

    def __init__(self, name: str, context: ot_span.SpanContext, **kwargs):
        kwargs['name'] = name
        kwargs['context'] = make_hashable(context)
        kwargs['attributes'] = kwargs.get('attributes', {}) or {}
        kwargs['links'] = kwargs.get('links', {}) or {}

        super().__init__(**kwargs)

    def end(self, end_time: Optional[TTimestamp] = None):
        """See [end][opentelemetry.trace.span.Span.end]"""
        if end_time:
            self.end_timestamp = end_time
        else:
            self.end_timestamp = time.time_ns()

        self.status = trace_status.StatusCode.OK

    def get_span_context(self) -> ot_span.SpanContext:
        """See [end][opentelemetry.trace.span.Span.get_span_context]"""

        return self.context

    def set_attributes(self, attributes: Dict[str, ot_types.AttributeValue]) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attributes]"""

        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: ot_types.AttributeValue) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attribute]"""

        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.add_event]"""
        self.events.append((name, attributes, timestamp or time.time_ns()))

    def add_link(
        self,
        context: ot_span.SpanContext,
        attributes: ot_types.Attributes = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.add_link]"""

        context = make_hashable(context)

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def update_name(self, name: str) -> None:
        """See [end][opentelemetry.trace.span.Span.update_name]."""

        self.name = name

    def is_recording(self) -> bool:
        """See [end][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_status.StatusCode.UNSET

    def set_status(
        self,
        status: Union[ot_span.Status, ot_span.StatusCode],
        description: Optional[str] = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.set_status]"""

        if isinstance(status, ot_span.Status):
            if description is not None:
                raise ValueError("Ambiguous status description provided both in `status.description` and in `description`.")
            
            self.status = status.status_code
            self.status_description = status.description
        else:
            self.status = status
            self.status_description = description

    def record_exception(
        self,
        exception: Exception,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
        escaped: bool = False
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.record_exception]"""

        self.status = trace_status.StatusCode.ERROR

        self.add_event(
            self._attr("exception"),
            attributes,
            timestamp
        )

class DictNamespace(Dict[str, ot_types.AttributeValue]):
    """View into a dict with keys prefixed by some `namespace` string.
    
    Replicates the values without the prefix in self.
    """

    def __init__(self, parent: Dict, namespace: str, **kwargs):
        self.parent = parent
        self.namespace = namespace

    def __getitem__(self, key):
        return dict.__getitem__(self, key)
    
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.parent[f"{self.namespace}.{key}"] = value

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        del self.parent[f"{self.namespace}.{key}"]

class Span(OTSpan):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    @property
    def span_id(self) -> TSpanID:
        """Identifier for the span."""

        return self.context.span_id

    @property
    def trace_id(self) -> TTraceID:
        """Identifier for the trace this span belongs to."""

        return self.context.trace_id

    @functools.cached_property
    def parent_context(self) -> Optional[HashableSpanContext]:
        """Context of parent span if any.

        None if this is a root span.
        """

        for link_context, link_attributes in self.links.items():
            if link_attributes.get(self._attr("relationship")) == "parent":
                return link_context

        return None

    @functools.cached_property
    def parent_span_id(self) -> Optional[TSpanID]:
        """Id of parent span if any.

        None if this is a root span.
        """

        parent_context = self.parent_context
        if parent_context is not None:
            return parent_context.span_id

        return None

    @property
    def tags(self) -> List[str]:
        """Tags associated with the span."""

        return self.attributes.get(self._attr("tags"), [])
    @tags.setter
    def tags(self, value: List[str]):
        self.attributes[self.attr("tags")] = value

    @property
    def span_type(self) -> SpanType:
        """Type of span."""

        return SpanType(self.attributes.get(self._attr("span_type"), SpanType.UNTYPED.name))

    @span_type.setter
    def span_type(self, value: SpanType):
        self.attributes[self._attr("span_type")] = value

    attributes_metadata: DictNamespace[str, ot_types.AttributeValue] 
    # will be set as a DictNamespace indexing elements in attributes
    @property
    def metadata(self) -> DictNamespace[str, ot_types.AttributeValue]:
        return self.attributes_metadata

    @metadata.setter
    def metadata(self, value: Dict[str, str]):
        for k, v in value.items():
            self.attributes_metadata[k] = v

    # input: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property
    # output: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property

    def __init__(self, **kwargs):
        kwargs['attributes_metadata'] = DictNamespace(parent={}, namespace="temp")
        # Temporary fake for validation in super.__init__ below.

        super().__init__(**kwargs)

        # Actual. This is needed as pydantic will copy attributes dict in init.
        self.attributes_metadata = DictNamespace(
            parent=self.attributes,
            namespace=self._attr("metadata")
        )

        self.set_attribute(self._attr("span_type"), self.__class__.__name__)

    @staticmethod
    def attribute_property(name: str, typ: Type):
        """A property that will get/set values from self.attributes."""

        def getter(self):
            return self.attributes.get(self._attr(name))
        
        def setter(self, value):
            if not isinstance(value, typ):
                raise ValueError(f"Expected {typ} for {self.name} but got {type(value)}.")

            self.attributes[self._attr(name)] = value

        return property(getter, setter)

class Tracer(pydantic.BaseModel, ot_trace.Tracer):
    """Implementation of OpenTelemetry Tracer requirements."""

    stack: contextvars.ContextVar[HashableSpanContext] = pydantic.Field(
        default_factory=lambda: contextvars.ContextVar("stack", default=None),
        exclude=True
    )

    instrumenting_module_name: str = "trulens_eval"
    instrumenting_library_version: Optional[str] = None#trulens_eval.__version__

    spans: ContextMapping[
        Mapping[str, ot_types.AttributeValue],
    ] = pydantic.Field(default_factory=dict)
    """Spans recorded by the tracer."""

    trace_id: TTraceID
    """Unique identifier for the trace."""

    model_config = {
        'arbitrary_types_allowed': True,
        'use_attribute_docstrings': True
    }
    """Pydantic configuration."""

    def __init__(self, **kwargs):
        trace_id = random.getrandbits(NumTraceIDBits)

        kwargs['trace_id'] = trace_id

        super().__init__(**kwargs)

    def start_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_trace.types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        """See [start_span][opentelemetry.trace.Tracer.start_span]."""

        if context is None:
            parent_context = self.stack.get()

        else:
            parent_context = make_hashable(context)

            if parent_context.trace_id != self.trace_id:
                logger.warning("Parent context is not being traced by this tracer.")

        span_context = HashableSpanContext(
            trace_id=self.trace_id,
            span_id=random.getrandbits(NumSpanIDBits),
            is_remote=False
        )

        span = SpanUntyped(
            name=name,
            context=span_context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception
        )

        if parent_context is not None:
            span.add_link(parent_context, {Span._attr("relationship"): "parent"})

        self.spans[span_context] = span

        return span

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = opentelemetry.trace.SpanKind.INTERNAL,
        attributes: ot_types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[OTSpan]:
        """See [start_as_current_span][opentelemetry.trace.Tracer.start_as_current_span]."""

        if context is not None:
            print("start_as_current_span", context)
            context = make_hashable(context)

        span = self.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception
        )

        token = self.stack.set(span.context)

        # Unsure if this ot_trace stuff is needed.
        span_token = ot_trace.use_span(span, end_on_exit=end_on_exit).__enter__()
        yield span

        # Same
        span_token.__exit__(None, None, None)

        self.stack.reset(token)
        return

class SpanUntyped(Span):
    """Generic span type.
    
    This represents spans that are being recorded but have not yet been
    determined to be of a particular type.
    """

    pass

class SpanRoot(Span):
    """A root span encompassing some collection of spans.

    Does not indicate any particular activity by itself beyond its children.
    """

    def parent_context(self):
        raise ValueError("Root span has no parent context.")

    def parent_span_id(self):
        raise ValueError("Root span has no parent span id.")

class SpanRetriever(Span):
    """A retrieval."""

    input_text = Span.attribute_property("input_text", str)
    """Input text whose related contexts are being retrieved."""

    input_embedding = Span.attribute_property("input_embedding", List[float])
    """Embedding of the input text."""

    distance_type = Span.attribute_property("distance_type", str)
    """Distance function used for ranking contexts."""

    num_contexts = Span.attribute_property("num_contexts", int)
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = Span.attribute_property("retrieved_contexts", List[str])
    """The retrieved contexts."""

class SpanReranker(Span):
    pass

class SpanLLM(Span):
    pass

class SpanEmbedding(Span):
    pass

class SpanTool(Span):
    pass

class SpanAgent(Span):
    pass

class SpanTask(Span):
    pass

class SpanOther(Span):
    pass

class SpanType(Enum):
    """Span types.
    
    This is a bit redundant with the span type hierarchy above. It is here for
    convenience of looking up types in means other than `__class__` or via
    `isinstance`.
    """

    UNTYPED = SpanUntyped.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanUntyped]."""

    ROOT = SpanRoot.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanRoot]."""

    RETRIEVER = SpanRetriever.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanRetriever]."""

    RERANKER = SpanReranker.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanReranker]."""

    LLM = SpanLLM.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanLLM]."""

    EMBEDDING = SpanEmbedding.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanEmbedding]."""

    TOOL = SpanTool.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanTool]."""

    AGENT = SpanAgent.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanAgent]."""

    TASK = SpanTask.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanTask]."""

    OTHER = SpanOther.__name__
    """See [SpanUntyped][trulens_eval.schema.span.SpanOther]."""
