"""Tracing and spans.

This module contains a [OTSpan][trulens_eval.trace.OTSpan], a Span
implementation conforming to the OpenTelemetry span specification and related
utilities. These are further specialized in
[Span][trulens_eval.trace.span.Span].
"""

from __future__ import annotations

from logging import getLogger
import time
from typing import (ClassVar, Dict, List, Mapping, Optional, Tuple, TypeVar,
                    Union)

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.trace import status as trace_status
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
import pydantic
from pydantic import PlainSerializer
from pydantic.functional_validators import PlainValidator
from typing_extensions import Annotated
from typing_extensions import TypeAliasType

logger = getLogger(__name__)

# Type alises

A = TypeVar("A")
B = TypeVar("B")

TTimestamp = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NUM_TIMESTAMP_BITS = 64

TSpanID = int
"""Type of span identifiers.

64 bit int as per OpenTelemetry.
"""
NUM_SPANID_BITS = 64
"""Number of bits in a span identifier."""

TTraceID = int
"""Type of trace identifiers.

128 bit int as per OpenTelemetry.
"""
NUM_TRACEID_BITS = 128
"""Number of bits in a trace identifier."""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue", 
    Union[
        str, int, float, bool,
        List['TLensedAttributeValue'], Dict[str, 'TLensedAttributeValue']
    ]
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes = Dict[str, TLensedAttributeValue]

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

def deserialize_contextmapping(
    v: List[Tuple[HashableSpanContext, T]]
) -> Dict[HashableSpanContext, T]:
    """Deserialize a list of tuples as a dictionary."""

    # skip last element of SpanContext as it is computed in SpanContext.__init__ from others
    return {HashableSpanContext(*k[0:5]): v for k, v in v} 

def serialize_contextmapping(
    v: Dict[HashableSpanContext, T],
) -> List[Tuple[HashableSpanContext, T]]:
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


def make_hashable(
    context: Optional[ot_span.SpanContext]
) -> Optional[HashableSpanContext]:
    """Make a SpanContext a HashableSpanContext.
    
    Returns None if input is None.
    """
    # HACK015: replacing class of contexts to add hashing

    if context is None:
        return None

    if context.__class__ is not HashableSpanContext:
        context.__class__ = HashableSpanContext

    # Return not needed but useful for type checker.
    return context


class OTSpan(pydantic.BaseModel, ot_span.Span):
    """Implementation of OpenTelemetry Span requirements.
    
    See also [OpenTelemetry
    Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span) and
    [OpenTelemetry Span
    specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md).

    There are a few places where this class deviates from OT specification.
    However, the conversion to
    [ReadableSpan][opentelemetry.sdk.trace.ReadableSpan] with
    [freeze][trulens_eval.trace.OTSpan.freeze] produces to-spec representations.
    The deviations are:

    - contexts are stored as
      [HashableSpanContext][trulens_eval.trace.HashableSpanContext] instead
      of [SpanContext][opentelemetry.trace.span.SpanContext].

    - attributes are not strictly as per spec which does not allow recursive
      types. They are instead json-like which allows storing recursive
      unstructured type. When freezing with
      [freeze][trulens_eval.trace.OTSpan.freeze], non-spec values are mapped to
      multiple in-spec attributes. The reverse is done when thawing with
      [thaw][trulens_eval.trace.OTSpan.thaw].
    """

    _vendor: ClassVar[str] = "trulens_eval"
    """Vendor name as per OpenTelemetry attribute keys specifications."""

    @classmethod
    def vendor_attr(cls, name: str) -> str:
        """Add vendor prefix to attribute name."""

        return f"{cls._vendor}@{name}"

    model_config = {
        'arbitrary_types_allowed': True,
        'use_attribute_docstrings': True
    }
    """Pydantic configuration."""

    name: str
    """Name of span."""

    kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL
    """Kind of span."""

    parent_context: Optional[HashableSpanContext] = None
    """Parent span context.
    
    None if no parent.
    """

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

    events: List[Tuple[str, ot_types.Attributes, TTimestamp]] = \
        pydantic.Field(default_factory=list)
    """Events recorded in the span.
    
    !!! Warning

        Events in OpenTelemetry seem to be synonymous to logs. Do not store
        anything we want to query or process in events.
    """

    links: ContextMapping[Mapping[str, ot_types.AttributeValue]] = \
        pydantic.Field(default_factory=dict)
    """Relationships to other spans with attributes on each link."""

    attributes: TLensedAttributes = \
        pydantic.Field(default_factory=dict)
    """Attributes of span."""

    def freeze(self) -> otsdk_trace.ReadableSpan:
        """Freeze the span into a [ReadableSpan][opentelemetry.sdk.trace.ReadableSpan].
    
        These are the views of spans that are exported by exporters.
        """

        return otsdk_trace.ReadableSpan(
            name=self.name,
            context=self.context,
            parent=self.parent_context,
            resource = None,
            attributes=self.attributes,
            events=self.events,
            links=self.links,
            kind=self.kind,
            instrumentation_info = None,
            status=self.status,
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            instrumentation_scope = None
        )

    @classmethod
    def thaw(cls, span: otsdk_trace.ReadableSpan) -> OTSpan:
        """Import a [ReadableSpan][opentelemetry.sdk.trace.ReadableSpan] as a
        [OTSpan][trulens_eval.trace.OTSpan]."""

        return cls(
            name=span.name,
            context=make_hashable(span.context),
            parent_context=make_hashable(span.parent),
            attributes=span.attributes,
            events=span.events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_timestamp=span.start_time,
            end_timestamp=span.end_time
        )

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
        """See [get_span_context][opentelemetry.trace.span.Span.get_span_context]."""

        return self.context

    def set_attributes(self, attributes: Dict[str, ot_types.AttributeValue]) -> None:
        """See [set_attributes][opentelemetry.trace.span.Span.set_attributes]."""

        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: ot_types.AttributeValue) -> None:
        """See [set_attribute][opentelemetry.trace.span.Span.set_attribute]."""

        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None
    ) -> None:
        """See [add_event][opentelemetry.trace.span.Span.add_event]."""

        self.events.append((name, attributes, timestamp or time.time_ns()))

    def add_link(
        self,
        context: ot_span.SpanContext,
        attributes: ot_types.Attributes = None
    ) -> None:
        """See [add_link][opentelemetry.trace.span.Span.add_link]."""

        context = make_hashable(context)

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def update_name(self, name: str) -> None:
        """See [update_name][opentelemetry.trace.span.Span.update_name]."""

        self.name = name

    def is_recording(self) -> bool:
        """See [is_recording][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_status.StatusCode.UNSET

    def set_status(
        self,
        status: Union[ot_span.Status, ot_span.StatusCode],
        description: Optional[str] = None
    ) -> None:
        """See [set_status][opentelemetry.trace.span.Span.set_status]."""

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
        """See [record_exception][opentelemetry.trace.span.Span.record_exception]."""

        self.status = trace_status.StatusCode.ERROR

        self.add_event(
            self.vendor_attr("exception"),
            attributes,
            timestamp
        )

    @property
    def span_id(self) -> TSpanID:
        """Identifier for the span."""

        return self.context.span_id

    @property
    def trace_id(self) -> TTraceID:
        """Identifier for the trace this span belongs to."""

        return self.context.trace_id

    # OTEL specificaiton is ambiguous on how parent relationship is to be
    # stored. The python api does not include a parent_context or other such
    # field but api as described in
    # https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md
    # does.
    @property
    def parent_span_id(self) -> Optional[TSpanID]:
        """Id of parent span if any."""

        parent_context = self.parent_context
        if parent_context is not None:
            return parent_context.span_id

        return None

    @property
    def parent_trace_id(self) -> Optional[TTraceID]:
        """Id of parent trace if any."""

        parent_context = self.parent_context
        if parent_context is not None:
            return parent_context.trace_id

        return None

OTSpan.update_forward_refs()