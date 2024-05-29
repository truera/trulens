"""Tracing and spans.

This module contains a [OTSpan][trulens_eval.trace.OTSpan], a Span
implementation conforming to the OpenTelemetry span specification and related
utilities. These are further specialized in
[Span][trulens_eval.trace.span.Span].
"""

from __future__ import annotations

from logging import getLogger
import time
from typing import (ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar,
                    Union)

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.trace import status as trace_status
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
import pydantic
from pydantic import PlainSerializer
from pydantic.functional_validators import PlainValidator
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Annotated
from typing_extensions import TypeAliasType

from trulens_eval.utils.python import NoneType
from trulens_eval.utils.serial import Lens

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

TLensedBaseType = Union[str, int, float, bool, NoneType]
"""Type of base types in span attributes.

!!! Warning

    OpenTelemetry does not allow None as an attribute value. However, we allow
    it in lensed attributes.
"""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue", 
    Union[
        str, int, float, bool,
        NoneType, # NOTE(piotrm): None is not technically allowed as an attribute value.
        Sequence['TLensedAttributeValue'],
        'TLensedAttributes'
    ]
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes = Dict[str, TLensedAttributeValue]

def inflate_lensed_mapping(
    m: ot_types.Attributes,
    prefix: str = "trulens_eval@"
) -> TLensedAttributes:
    """Inflate OpenTelemetry attributes into lensed attributes."""

    ret = {}

    for k, v in m.items():
        if not k.startswith(prefix):
            logger.warning("Attribute %s does not start with %s. Skipping.", k, prefix)
            continue

        path = Lens.of_string(k[len(prefix):])
        path.path[0].item_or_attribute = prefix + path.path[0].item_or_attribute
        ret = path.set(ret, v)

    return ret

def flatten_value(
    v: TLensedAttributeValue,
    path: Optional[Lens] = None
) -> Iterable[Tuple[Lens, ot_types.AttributeValue]]:

    """Flatten a lensed value into OpenTelemetry attribute values."""

    if path is None:
        path = Lens()

    #if v is None:
        # OpenTelemetry does not allow None as an attribute value. Unsure what
        # is best to do here. Returning "None" for now.
    #    yield (path, "None")

    elif isinstance(v, TLensedBaseType):
        yield (path, v)

    elif isinstance(v, Sequence) and all(isinstance(e, TLensedBaseType) for e in v):
        yield (path, v)

    elif isinstance(v, Sequence):
        for i, e in enumerate(v):
            yield from flatten_value(v=e, path=path[i])

    elif isinstance(v, Mapping):
        for k, e in v.items():
            yield from flatten_value(v=e, path=path[k])

    else:
        raise ValueError(f"Do not know how to flatten value of type {type(v)} to OTEL attributes.")

def flatten_lensed_attributes(
    m: TLensedAttributes,
    path: Optional[Lens] = None,
    prefix: str = "trulens_eval@"
) -> ot_types.Attributes:
    """Flatten lensed attributes into OpenTelemetry attributes."""

    if path is None:
        path = Lens()

    ret = {}
    for k, v in m.items():
        if k.startswith(prefix):
            # Only flattening those attributes that begin with `prefix` are
            # those are the ones coming from trulens_eval.
            for p, a in flatten_value(v, path[k]):
                ret[str(p)] = a
        else:
            ret[k] = v

    return ret

T = TypeVar("T")

class WithHashableSpanContext(): # must be mixed into SpanContext
    """Mixin into SpanContext that adds hashing.

    Does not change data layout or behaviour. Changing SpanContext
    `__class__` or `__bases__` to include this should be safe.
    """

    def __hash__(self: ot_span.SpanContext):
        return hash((self.trace_id, self.span_id))

    def __eq__(self: ot_span.SpanContext, other: ot_span.SpanContext):
        return self.trace_id == other.trace_id and self.span_id == other.span_id
    
    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, _handler) -> JsonSchemaValue:
        return { 
            'description': 'SpanContext that can be hashed.  Does not change data layout or behaviour. Changing SpanContext `__class__` with this should be safe.',
            'items': {
                'maxItems': 6,
                'minItems': 6,
                'prefixItems': [
                    {"type": "integer", "description": "The ID of the trace that this span belongs to." },
                    {"type": "integer", "description": "This span's ID." },
                    {"type": "boolean", "description": "True if propagated from a remote parent." },
                    {"type": "integer", "description": "Trace options to propagate. See the `W3C Trace Context - Traceparent`_ spec for details." },
                    {"type": "integer", "description": "A list of key-value pairs representing vendor-specific trace info. Keys and values are strings of up to 256 printable US-ASCII characters. Implementations should conform to the `W3C Trace Context - Tracestate` spec, which describes additional restrictions on valid field values." },
                    {"type": "boolean", "description": "True if the span context is valid." },
                ]
            },
            'title': 'SpanContext',
            'type': 'array'
        }

# HACK015: add hashing to contexts
# Patch span context to use the above hashing and schema defs.
ot_span.SpanContext.__bases__ = (WithHashableSpanContext, *ot_span.SpanContext.__bases__, )

def validate_contextmapping(
    v: List[Tuple[ot_span.SpanContext, T]]
) -> Dict[ot_span.SpanContext, T]:
    """Deserialize a list of tuples as a dictionary."""

    if isinstance(v, dict):
        # Already a dict.
        return v

    # skip last element of SpanContext as it is computed in SpanContext.__init__ from others
    return {ot_span.SpanContext(*k[0:5]): v for k, v in v} 

def serialize_contextmapping(
    v: Dict[ot_span.SpanContext, T],
) -> List[Tuple[ot_span.SpanContext, T]]:
    """Serialize a dictionary as a list of tuples."""

    return list(v.items())

ContextMapping = Annotated[
    Dict[ot_span.SpanContext, T],
    PlainSerializer(serialize_contextmapping),
    PlainValidator(validate_contextmapping)
]
"""Type annotation for pydantic fields that store dictionaries whose keys are
(Hashable) SpanContext.

This is needed to help pydantic figure out how to serialize and deserialize these dicts.
"""


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

    parent_context: Optional[ot_span.SpanContext] = None
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

    context: ot_span.SpanContext
    """Unique immutable identifier for the span."""

    events: List[Tuple[str, ot_types.Attributes, TTimestamp]] = \
        pydantic.Field(default_factory=list)
    """Events recorded in the span.
    
    !!! Warning

        Events in OpenTelemetry seem to be synonymous to logs. Do not store
        anything we want to query or process in events.
    """

    links: ContextMapping[TLensedAttributes] = \
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
            attributes=flatten_lensed_attributes(self.attributes),
            events=self.events,
            links=[ot_trace.Link(context=c, attributes=a) for c, a in self.links.items()],
            kind=self.kind,
            instrumentation_info = None,
            status=ot_span.Status(self.status, self.status_description),
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
            context=span.context,
            parent_context=span.parent,
            attributes=inflate_lensed_mapping(span.attributes),
            events=span.events,
            links={link.context: inflate_lensed_mapping(link.attributes) for link in span.links},
            kind=span.kind,
            status=span.status.status_code,
            status_description=span.status.description,
            start_timestamp=span.start_time,
            end_timestamp=span.end_time
        )

    def __init__(self, name: str, context: ot_span.SpanContext, **kwargs):
        kwargs['name'] = name
        kwargs['context'] = context
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