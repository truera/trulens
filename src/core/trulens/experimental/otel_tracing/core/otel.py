from __future__ import annotations

import contextlib
import contextvars
import logging
import random
import time
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeAliasType,
    TypeVar,
    Union,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk import resources as resources_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.util import types as types_api
import pydantic
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils.serial import Lens

logger = logging.getLogger(__name__)

# Type alises

A = TypeVar("A")
B = TypeVar("B")

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

TTimestamp = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NUM_TIMESTAMP_BITS = 64

TLensedBaseType = Union[str, int, float, bool]
"""Type of base types in span attributes.
!!! Warning
    OpenTelemetry does not allow None as an attribute value. However, we allow
    it in lensed attributes.
"""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue",
    Union[
        str,
        int,
        float,
        bool,
        python_utils.NoneType,  # NOTE(piotrm): None is not technically allowed as an attribute value.
        Sequence["TLensedAttributeValue"],
        "TLensedAttributes",
    ],
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes = Dict[str, TLensedAttributeValue]


def flatten_value(
    v: TLensedAttributeValue, path: Optional[Lens] = None
) -> Iterable[Tuple[Lens, types_api.AttributeValue]]:
    """Flatten a lensed value into OpenTelemetry attribute values."""

    if path is None:
        path = Lens()

    # if v is None:
    # OpenTelemetry does not allow None as an attribute value. Unsure what
    # is best to do here. Returning "None" for now.
    #    yield (path, "None")

    elif v is None:
        pass

    elif isinstance(v, TLensedBaseType):
        yield (path, v)

    elif isinstance(v, Sequence) and all(
        isinstance(e, TLensedBaseType) for e in v
    ):
        yield (path, v)

    elif isinstance(v, Sequence):
        for i, e in enumerate(v):
            yield from flatten_value(v=e, path=path[i])

    elif isinstance(v, Mapping):
        for k, e in v.items():
            yield from flatten_value(v=e, path=path[k])

    else:
        raise ValueError(
            f"Do not know how to flatten value of type {type(v)} to OTEL attributes."
        )


def flatten_lensed_attributes(
    m: TLensedAttributes, path: Optional[Lens] = None, prefix: str = ""
) -> types_api.Attributes:
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


def new_trace_id():
    return int(random.getrandbits(128) & trace_api.span._TRACE_ID_MAX_VALUE)


def new_span_id():
    return int(random.getrandbits(64) & trace_api.span._SPAN_ID_MAX_VALUE)


class TraceState(serial_utils.SerialModel, trace_api.span.TraceState):
    """OTEL [TraceState][opentelemetry.trace.TraceState] requirements."""

    # Hackish: trace_api.span.TraceState uses _dict internally.
    _dict: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)


class SpanContext(serial_utils.SerialModel):
    """OTEL [SpanContext][opentelemetry.trace.SpanContext] requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    trace_id: int = pydantic.Field(default_factory=new_trace_id)
    """Unique identifier for the trace.

    Each root span has a unique trace id."""

    span_id: int = pydantic.Field(default_factory=new_span_id)
    """Identifier for the span.

    Meant to be at least unique within the same trace_id.
    """

    trace_flags: trace_api.TraceFlags = pydantic.Field(
        trace_api.DEFAULT_TRACE_OPTIONS
    )

    trace_state: TraceState = pydantic.Field(default_factory=TraceState)

    is_remote: bool = False

    _tracer: Tracer = pydantic.PrivateAttr(None)

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)


def lens_of_flat_key(key: str) -> serial_utils.Lens:
    """Convert a flat dict key to a lens."""
    lens = serial_utils.Lens()
    for step in key.split("."):
        lens = lens[step]

    return lens


class Span(serial_utils.SerialModel, trace_api.Span):
    """OTEL [Span][opentelemetry.trace.Span] requirements.

    See also [OpenTelemetry
    Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span) and
    [OpenTelemetry Span
    specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _name: str = pydantic.PrivateAttr(None)

    @property
    def name(self) -> str:
        """Name of the span."""

        return self._name

    _kind: trace_api.SpanKind = pydantic.PrivateAttr(
        trace_api.SpanKind.INTERNAL
    )

    @property
    def kind(self) -> trace_api.SpanKind:
        """Kind of span."""

        return self._kind

    _context: SpanContext = pydantic.PrivateAttr(None)

    @property
    def context(self) -> SpanContext:
        """Trace/span identifiers."""

        return self._context

    _parent: Optional[SpanContext] = pydantic.PrivateAttr(None)

    @property
    def parent(self) -> Optional[SpanContext]:
        """Optional parent identifiers."""

        return self._parent

    _status: trace_api.status.StatusCode = pydantic.PrivateAttr(
        default=trace_api.status.StatusCode.UNSET
    )

    @property
    def status(self) -> trace_api.status.StatusCode:
        """Status of the span as per OpenTelemetry Span requirements."""

        return self._status

    _status_description: Optional[str] = pydantic.PrivateAttr(None)

    @property
    def status_description(self) -> Optional[str]:
        """Status description as per OpenTelemetry Span requirements."""

        return self._status_description

    _events: List[Tuple[str, trace_api.types.Attributes, TTimestamp]] = (
        pydantic.PrivateAttr(default_factory=list)
    )

    @property
    def events(
        self,
    ) -> List[Tuple[str, trace_api.types.Attributes, TTimestamp]]:
        """Events recorded in the span.

        !!! Warning

            Events in OpenTelemetry seem to be synonymous to logs. Do not store
            anything we want to query or process in events.
        """
        return self._events

    _links: Dict[SpanContext, trace_api.types.Attributes] = (
        pydantic.PrivateAttr(default_factory=dict)
    )

    @property
    def links(self) -> trace_api._Links:
        """Relationships to other spans with attributes on each link."""

        return self._links

    _attributes: trace_api.types.Attributes = pydantic.PrivateAttr(
        default_factory=dict
    )

    @property
    def attributes(self) -> trace_api.types.Attributes:
        return self._attributes

    _start_timestamp: int = pydantic.PrivateAttr(default_factory=time.time_ns)

    @property
    def start_timestamp(self) -> int:
        """Start time in nanoseconds since epoch."""

        return self._start_timestamp

    _end_timestamp: Optional[int] = pydantic.PrivateAttr(None)

    @property
    def end_timestamp(self) -> Optional[int]:
        """End time in nanoseconds since epoch. None if not yet finished."""

        return self._end_timestamp

    _record_exception: bool = pydantic.PrivateAttr(True)

    _set_status_on_exception: bool = pydantic.PrivateAttr(True)

    _tracer: Tracer = pydantic.PrivateAttr(None)

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)

        if self._start_timestamp is None:
            self._start_timestamp = time.time_ns()

    def update_name(self, name: str) -> None:
        """See [update_name][opentelemetry.trace.span.Span.update_name]."""

        self._name = name

    def get_span_context(self) -> trace_api.span.SpanContext:
        """See [get_span_context][opentelemetry.trace.span.Span.get_span_context]."""

        return self.context

    def set_status(
        self,
        status: Union[trace_api.span.Status, trace_api.span.StatusCode],
        description: Optional[str] = None,
    ) -> None:
        """See [set_status][opentelemetry.trace.span.Span.set_status]."""

        if isinstance(status, trace_api.span.Status):
            if description is not None:
                raise ValueError(
                    "Ambiguous status description provided both in `status.description` and in `description`."
                )

            self._status = status.status_code
            self._status_description = status.description
        else:
            self._status = status
            self._status_description = description

    def add_event(
        self,
        name: str,
        attributes: types_api.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
    ) -> None:
        """See [add_event][opentelemetry.trace.span.Span.add_event]."""

        self._events.append((name, attributes, timestamp or time.time_ns()))

    def add_link(
        self,
        context: trace_api.span.SpanContext,
        attributes: types_api.Attributes = None,
    ) -> None:
        """See [add_link][opentelemetry.trace.span.Span.add_link]."""

        if attributes is None:
            attributes = {}

        self._links[context] = attributes

    def is_recording(self) -> bool:
        """See [is_recording][opentelemetry.trace.span.Span.is_recording]."""

        return self._status == trace_api.status.StatusCode.UNSET

    def set_attributes(
        self, attributes: Dict[str, types_api.AttributeValue]
    ) -> None:
        """See [set_attributes][opentelemetry.trace.span.Span.set_attributes]."""

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_attribute(self, key: str, value: types_api.AttributeValue) -> None:
        """See [set_attribute][opentelemetry.trace.span.Span.set_attribute]."""

        self.attributes[key] = value

    def record_exception(
        self,
        exception: BaseException,
        attributes: types_api.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
        escaped: bool = False,
    ) -> None:
        """See [record_exception][opentelemetry.trace.span.Span.record_exception]."""

        if timestamp is None:
            timestamp = time.time_ns()

        if self._set_status_on_exception:
            self._end_timestamp = time.time_ns()
            self._status = trace_api.status.StatusCode.ERROR

        if self._record_exception:
            if attributes is None:
                attributes = {}

            attributes["exc_type"] = python_utils.class_name(type(exception))
            attributes["exc_val"] = str(exception)
            if exception.__traceback__ is not None:
                attributes["code_line"] = python_utils.code_line(
                    exception.__traceback__.tb_frame, show_source=True
                )

            self.add_event("trulens.exception", attributes, timestamp)

    def end(self, end_time: Optional[TTimestamp] = None):
        """See [end][opentelemetry.trace.span.Span.end]"""

        if end_time is None:
            end_time = time.time_ns()

        self._status = trace_api.status.StatusCode.OK
        self._end_timestamp = end_time

    def __enter__(self) -> Span:
        """See [__enter__][opentelemetry.trace.span.Span.__enter__]."""

        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """See [__exit__][opentelemetry.trace.span.Span.__exit__]."""

        if exc_val is not None:
            self.record_exception(exception=exc_val)
            raise exc_val

        self.end()

    # Rest of these methods are for exporting spans to ReadableSpan

    @staticmethod
    def otel_context_of_context(context: SpanContext) -> trace_api.SpanContext:
        return trace_api.SpanContext(
            trace_id=context.trace_id,
            span_id=context.span_id,
            is_remote=False,
        )

    def otel_name(self) -> str:
        return self.name

    def otel_context(self) -> types_api.SpanContext:
        return self.otel_context_of_context(self.context)

    def otel_parent_context(self) -> Optional[types_api.SpanContext]:
        if self.parent is None:
            return None

        return self.otel_context_of_context(self.parent)

    def otel_attributes(self) -> types_api.Attributes:
        return flatten_lensed_attributes(self.attributes)

    def otel_kind(self) -> types_api.SpanKind:
        return trace_api.SpanKind.INTERNAL

    def otel_status(self) -> trace_api.status.Status:
        return trace_api.status.Status(self.status, self.status_description)

    def otel_resource_attributes(self) -> Dict[str, Any]:
        return {
            "service.namespace": "trulens",
        }

    def otel_resource(self) -> resources_sdk.Resource:
        return resources_sdk.Resource(
            attributes=self.otel_resource_attributes()
        )

    def otel_events(self) -> List[types_api.Event]:
        return self.events

    def otel_links(self) -> List[types_api.Link]:
        return self.links

    def otel_freeze(self) -> trace_sdk.ReadableSpan:
        """Convert span to an OTEL compatible span for exporting to OTEL collectors.

        !!! Warning
            This is an experimental feature. OTEL integration is ongoing.
        """

        return trace_sdk.ReadableSpan(
            name=self.otel_name(),
            context=self.otel_context(),
            parent=self.otel_parent_context(),
            resource=self.otel_resource(),
            attributes=self.otel_attributes(),
            events=self.otel_events(),
            links=self.otel_links(),
            kind=self.otel_kind(),
            instrumentation_info=None,  # TODO
            status=self.otel_status(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            instrumentation_scope=None,  # TODO
        )


class Tracer(serial_utils.SerialModel, trace_api.Tracer):
    """OTEL [Tracer][opentelemetry.trace.Tracer] requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _instrumenting_module_name: Optional[str] = pydantic.PrivateAttr(None)
    """Name of the library/module that is instrumenting the code."""

    _instrumenting_library_version: Optional[str] = pydantic.PrivateAttr(None)
    """Version of the library that is instrumenting the code."""

    _tracer_provider: TracerProvider = pydantic.PrivateAttr(None)

    _span_class: Type[Span] = pydantic.PrivateAttr(Span)

    _span_context_class: Type[SpanContext] = pydantic.PrivateAttr(SpanContext)

    _attributes: Optional[trace_api.types.Attributes] = pydantic.PrivateAttr(
        None
    )

    _schema_url: Optional[str] = pydantic.PrivateAttr(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)

    @property
    def context_cvar(
        self,
    ) -> contextvars.ContextVar[context_api.context.Context]:
        return self._tracer_provider.context_cvar

    @property
    def trace_id(self) -> int:
        return self._tracer_provider.trace_id

    def start_span(
        self,
        name: Optional[str] = None,
        *,
        context: Optional[context_api.context.Context] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.types.Attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        cls: Optional[Type[Span]] = None,  # non-standard
    ) -> Span:
        if context is None:
            parent_context = self.context_cvar.get()

        else:
            if len(context) != 1:
                raise ValueError("Only one context span is allowed.")
            parent_span_encoding = next(iter(context.values()))

            parent_context = self._span_context_class(
                trace_id=parent_span_encoding.trace_id,
                span_id=parent_span_encoding.span_id,
                _tracer=self,
            )

        new_context = self._span_context_class(
            trace_id=self.trace_id, _tracer=self
        )

        if name is None:
            name = python_utils.class_name(self._span_class)

        if attributes is None:
            attributes = {}

        if self._attributes is not None:
            attributes.update(self._attributes)

        if cls is None:
            cls = self._span_class

        new_span = cls(
            _name=name,
            _context=new_context,
            _parent=parent_context,
            _kind=kind,
            _attributes=attributes,
            _links=links,
            _start_timestamp=start_time,
            _record_exception=record_exception,
            _set_status_on_exception=set_status_on_exception,
            _tracer=self,
        )

        return new_span

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: Optional[str] = None,
        context: Optional[trace_api.context.Context] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.types.Attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ):
        span = self.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )

        token = self.context_cvar.set(span.context)

        try:
            yield span

        except BaseException as e:
            if record_exception:
                span.record_exception(e)

        finally:
            self.context_cvar.reset(token)

            if end_on_exit:
                span.end()


class TracerProvider(serial_utils.SerialModel, trace_api.TracerProvider):
    """OTEL See [TracerProvider][opentelemetry.trace.TracerProvider]
    requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(Tracer)

    _context_cvar: contextvars.ContextVar[context_api.context.Context] = (
        pydantic.PrivateAttr(
            default_factory=lambda: contextvars.ContextVar(
                "context", default=None
            )
        )
    )

    @property
    def context_cvar(
        self,
    ) -> contextvars.ContextVar[context_api.context.Context]:
        return self._context_cvar

    _trace_id: int = pydantic.PrivateAttr(default_factory=new_trace_id)

    @property
    def trace_id(self) -> int:
        return self._trace_id

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[types_api.Attributes] = None,
    ):
        tracer = self._tracer_class(
            _instrumenting_module_name=instrumenting_module_name,
            _instrumenting_library_version=instrumenting_library_version,
            _attributes=attributes,
            _schema_url=schema_url,
            _tracer_provider=self,
        )

        return tracer
