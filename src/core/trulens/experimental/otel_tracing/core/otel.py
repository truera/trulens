# ruff: noqa: E402

"""OTEL Compatibility Classes

This module contains classes to support interacting with the OTEL ecosystem.
Additions on top of these meant for TruLens uses outside of OTEL compatibility
are found in `traces.py`.
"""

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
    TypeVar,
    Union,
)

import pydantic
from trulens.core._utils.pycompat import NoneType  # import style exception
from trulens.core._utils.pycompat import TypeAlias  # import style exception
from trulens.core._utils.pycompat import TypeAliasType  # import style exception
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing import _feature

_feature._FeatureSetup.assert_optionals_installed()  # checks to make sure otel is installed

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk import resources as resources_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.util import types as types_api

logger = logging.getLogger(__name__)

# Type alises

A = TypeVar("A")
B = TypeVar("B")

TSpanID: TypeAlias = int
"""Type of span identifiers.
64 bit int as per OpenTelemetry.
"""
NUM_SPANID_BITS: int = 64
"""Number of bits in a span identifier."""

TTraceID: TypeAlias = int
"""Type of trace identifiers.
128 bit int as per OpenTelemetry.
"""
NUM_TRACEID_BITS: int = 128
"""Number of bits in a trace identifier."""

TTimestamp: TypeAlias = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NUM_TIMESTAMP_BITS = 64

TLensedBaseType: TypeAlias = Union[str, int, float, bool]
"""Type of base types in span attributes.

!!! Warning
    OpenTelemetry does not allow None as an attribute value. Handling None is to
    be decided.
"""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue",
    Union[
        str,
        int,
        float,
        bool,
        NoneType,  # TODO(SNOW-1711929): None is not technically allowed as an attribute value.
        Sequence["TLensedAttributeValue"],  # type: ignore
        "TLensedAttributes",
    ],
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes: TypeAlias = Dict[str, TLensedAttributeValue]
"""Attribute dictionaries.

Note that this deviates from what OTEL allows as attribute values. Because OTEL
does not allow general recursive values to be stored as attributes, we employ a
system of flattening values before exporting to OTEL. In this process we encode
a single generic value as multiple attributes where the attribute name include
paths/lenses to the parts of the generic value they are representing. For
example, an attribute/value like `{"a": {"b": 1, "c": 2}}` would be encoded as
`{"a.b": 1, "a.c": 2}`. This process is implemented in the
`flatten_lensed_attributes` method.
"""


def flatten_value(
    v: TLensedAttributeValue, lens: Optional[serial_utils.Lens] = None
) -> Iterable[Tuple[serial_utils.Lens, types_api.AttributeValue]]:
    """Flatten recursive value into OTEL-compatible attribute values.

    See `TLensedAttributes` for more details.
    """

    if lens is None:
        lens = serial_utils.Lens()

    # TODO(SNOW-1711929): OpenTelemetry does not allow None as an attribute
    # value. Unsure what is best to do here.

    # if v is None:
    #    yield (path, "None")

    elif v is None:
        pass

    elif isinstance(v, TLensedBaseType):
        yield (lens, v)

    elif isinstance(v, Sequence) and all(
        isinstance(e, TLensedBaseType) for e in v
    ):
        yield (lens, v)

    elif isinstance(v, Sequence):
        for i, e in enumerate(v):
            yield from flatten_value(v=e, lens=lens[i])

    elif isinstance(v, Mapping):
        for k, e in v.items():
            yield from flatten_value(v=e, lens=lens[k])

    else:
        raise ValueError(
            f"Do not know how to flatten value of type {type(v)} to OTEL attributes."
        )


def flatten_lensed_attributes(
    m: TLensedAttributes,
    path: Optional[serial_utils.Lens] = None,
    prefix: str = "",
) -> types_api.Attributes:
    """Flatten lensed attributes into OpenTelemetry attributes."""

    if path is None:
        path = serial_utils.Lens()

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
    return int(
        random.getrandbits(NUM_TRACEID_BITS)
        & trace_api.span._TRACE_ID_MAX_VALUE
    )


def new_span_id():
    return int(
        random.getrandbits(NUM_SPANID_BITS) & trace_api.span._SPAN_ID_MAX_VALUE
    )


class TraceState(serial_utils.SerialModel, trace_api.span.TraceState):
    """[OTEL TraceState][opentelemetry.trace.TraceState] requirements."""

    # Hackish: trace_api.span.TraceState uses _dict internally.
    _dict: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)


class SpanContext(serial_utils.SerialModel):
    """[OTEL SpanContext][opentelemetry.trace.SpanContext] requirements."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # needed for enums that do not inherit from str
    )

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

    @pydantic.field_validator("trace_flags", mode="before")
    @classmethod
    def _validate_trace_flags(cls, v):
        """Validate trace flags.

        Pydantic does not seem to like classes that inherit from int without this.
        """
        return trace_api.TraceFlags(v)

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


class Span(
    pyschema_utils.WithClassInfo, serial_utils.SerialModel, trace_api.Span
):
    """[OTEL Span][opentelemetry.trace.Span] requirements.

    See also [OpenTelemetry
    Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span) and
    [OpenTelemetry Span
    specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md).
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # model_validate will fail without this
    )

    name: Optional[str] = None

    kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL

    context: SpanContext = pydantic.Field(default_factory=SpanContext)
    parent: Optional[SpanContext] = None

    status: trace_api.status.StatusCode = trace_api.status.StatusCode.UNSET
    status_description: Optional[str] = None

    events: List[Tuple[str, trace_api.types.Attributes, TTimestamp]] = (
        pydantic.Field(default_factory=list)
    )
    links: trace_api._Links = pydantic.Field(default_factory=dict)

    #    attributes: trace_api.types.Attributes = pydantic.Field(default_factory=dict)
    attributes: Dict = pydantic.Field(default_factory=dict)

    start_timestamp: int = pydantic.Field(default_factory=time.time_ns)

    end_timestamp: Optional[int] = None

    _record_exception: bool = pydantic.PrivateAttr(True)
    _set_status_on_exception: bool = pydantic.PrivateAttr(True)

    _tracer: Tracer = pydantic.PrivateAttr(None)
    """NON-STANDARD: The Tracer that produced this span."""

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def __init__(self, **kwargs):
        if kwargs.get("start_timestamp") is None:
            kwargs["start_timestamp"] = time.time_ns()

        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)

    def update_name(self, name: str) -> None:
        """See [OTEL update_name][opentelemetry.trace.span.Span.update_name]."""

        self.name = name

    def get_span_context(self) -> trace_api.span.SpanContext:
        """See [OTEL get_span_context][opentelemetry.trace.span.Span.get_span_context]."""

        return self.context

    def set_status(
        self,
        status: Union[trace_api.span.Status, trace_api.span.StatusCode],
        description: Optional[str] = None,
    ) -> None:
        """See [OTEL set_status][opentelemetry.trace.span.Span.set_status]."""

        if isinstance(status, trace_api.span.Status):
            if description is not None:
                raise ValueError(
                    "Ambiguous status description provided both in `status.description` and in `description`."
                )

            self.status = status.status_code
            self.status_description = status.description
        else:
            self.status = status
            self.status_description = description

    def add_event(
        self,
        name: str,
        attributes: types_api.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
    ) -> None:
        """See [OTEL add_event][opentelemetry.trace.span.Span.add_event]."""

        self.events.append((name, attributes, timestamp or time.time_ns()))

    def add_link(
        self,
        context: trace_api.span.SpanContext,
        attributes: types_api.Attributes = None,
    ) -> None:
        """See [OTEL add_link][opentelemetry.trace.span.Span.add_link]."""

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def is_recording(self) -> bool:
        """See [OTEL is_recording][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_api.status.StatusCode.UNSET

    def set_attributes(
        self, attributes: Dict[str, types_api.AttributeValue]
    ) -> None:
        """See [OTEL set_attributes][opentelemetry.trace.span.Span.set_attributes]."""

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_attribute(self, key: str, value: types_api.AttributeValue) -> None:
        """See [OTEL set_attribute][opentelemetry.trace.span.Span.set_attribute]."""

        self.attributes[key] = value

    def record_exception(
        self,
        exception: BaseException,
        attributes: types_api.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
        escaped: bool = False,  # purpose unknown
    ) -> None:
        """See [OTEL record_exception][opentelemetry.trace.span.Span.record_exception]."""

        if self._set_status_on_exception:
            self.set_status(
                trace_api.status.Status(trace_api.status.StatusCode.ERROR)
            )

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
        """See [OTEL end][opentelemetry.trace.span.Span.end]"""

        if end_time is None:
            end_time = time.time_ns()

        self.end_timestamp = end_time

        if self.is_recording():
            self.set_status(
                trace_api.status.Status(trace_api.status.StatusCode.OK)
            )

    def __enter__(self) -> Span:
        """See [OTEL __enter__][opentelemetry.trace.span.Span.__enter__]."""

        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """See [OTEL __exit__][opentelemetry.trace.span.Span.__exit__]."""

        try:
            if exc_val is not None:
                self.record_exception(exception=exc_val)
                raise exc_val
        finally:
            self.end()

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        return self.__exit__(exc_type, exc_val, exc_tb)

    # Rest of these methods are for exporting spans to ReadableSpan. All are not standard OTEL.

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
        #  TODO(SNOW-1711959)
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
        """Convert span to an OTEL compatible span for exporting to OTEL collectors."""

        return trace_sdk.ReadableSpan(
            name=self.otel_name(),
            context=self.otel_context(),
            parent=self.otel_parent_context(),
            resource=self.otel_resource(),
            attributes=self.otel_attributes(),
            events=self.otel_events(),
            links=self.otel_links(),
            kind=self.otel_kind(),
            instrumentation_info=None,  # TODO(SNOW-1711959)
            status=self.otel_status(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            instrumentation_scope=None,  # TODO(SNOW-1711959)
        )


class Tracer(serial_utils.SerialModel, trace_api.Tracer):
    """[OTEL Tracer][opentelemetry.trace.Tracer] requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _instrumenting_module_name: Optional[str] = pydantic.PrivateAttr(None)
    """Name of the library/module that is instrumenting the code."""

    _instrumenting_library_version: Optional[str] = pydantic.PrivateAttr(None)
    """Version of the library that is instrumenting the code."""

    _attributes: Optional[trace_api.types.Attributes] = pydantic.PrivateAttr(
        None
    )
    """Common attributes to add to all spans."""

    _schema_url: Optional[str] = pydantic.PrivateAttr(None)
    """Use unknown."""

    _tracer_provider: TracerProvider = pydantic.PrivateAttr(None)
    """NON-STANDARD: The TracerProvider that made this tracer."""

    _span_class: Type[Span] = pydantic.PrivateAttr(Span)
    """NON-STANDARD: The default span class to use when creating spans."""

    _span_context_class: Type[SpanContext] = pydantic.PrivateAttr(SpanContext)
    """NON-STANDARD: The default span context class to use when creating spans."""

    def __init__(self, _context: context_api.context.Context, **kwargs):
        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)

        self._context_cvar.set(_context)

    _context_cvar: contextvars.ContextVar[context_api.context.Context] = (
        pydantic.PrivateAttr(
            default_factory=lambda: contextvars.ContextVar(
                f"context_Tracer_{python_utils.context_id()}", default=None
            )
        )
    )

    @property
    def context_cvar(
        self,
    ) -> contextvars.ContextVar[context_api.context.Context]:
        """NON-STANDARD: The context variable to store the current span context."""

        return self._context_cvar

    @property
    def trace_id(self) -> int:
        return self._tracer_provider.trace_id

    def start_span(
        self,
        name: Optional[str] = None,
        *args,  # non-standard
        context: Optional[context_api.context.Context] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.types.Attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        cls: Optional[Type[Span]] = None,  # non-standard
        **kwargs,  # non-standard
    ) -> Span:
        """See [OTEL
        Tracer.start_span][opentelemetry.trace.Tracer.start_span]."""

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
            *args, trace_id=self.trace_id, _tracer=self, **kwargs
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
            name=name,
            context=new_context,
            parent=parent_context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_timestamp=start_time,
            _record_exception=record_exception,
            _status_on_exception=set_status_on_exception,
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
        """See [OTEL
        Tracer.start_as_current_span][opentelemetry.trace.Tracer.start_as_current_span]."""

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
    """[OTEL TracerProvider][opentelemetry.trace.TracerProvider]
    requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(Tracer)
    """NON-STANDARD: The default tracer class to use when creating tracers."""

    _context_cvar: contextvars.ContextVar[context_api.context.Context] = (
        pydantic.PrivateAttr(
            default_factory=lambda: contextvars.ContextVar(
                f"context_TracerProvider_{python_utils.context_id()}",
                default=None,
            )
        )
    )

    @property
    def context_cvar(
        self,
    ) -> contextvars.ContextVar[context_api.context.Context]:
        """NON-STANDARD: The context variable to store the current span context."""

        return self._context_cvar

    _trace_id: int = pydantic.PrivateAttr(default_factory=new_trace_id)

    @property
    def trace_id(self) -> int:
        """NON-STANDARD: The current trace id."""

        return self._trace_id

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[types_api.Attributes] = None,
    ):
        """See [OTEL
        TracerProvider.get_tracer][opentelemetry.trace.TracerProvider.get_tracer]."""

        tracer = self._tracer_class(
            _instrumenting_module_name=instrumenting_module_name,
            _instrumenting_library_version=instrumenting_library_version,
            _attributes=attributes,
            _schema_url=schema_url,
            _tracer_provider=self,
            _context=self.context_cvar.get(),
        )

        return tracer
