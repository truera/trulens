# ruff: noqa: E402

"""OTEL Compatibility Classes

This module contains classes to support interacting with the OTEL ecosystem.
Additions on top of these meant for TruLens uses outside of OTEL compatibility
are found in `span.py` and `trace.py`.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from types import TracebackType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing import _feature

_feature._FeatureSetup.assert_optionals_installed()  # checks to make sure otel is installed

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk import resources as resources_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.trace import span as span_api
from opentelemetry.util import types as types_api

logger = logging.getLogger(__name__)

# Type alises

A = TypeVar("A")
B = TypeVar("B")


class TraceState(serial_utils.SerialModel, span_api.TraceState):
    """[OTEL TraceState][opentelemetry.trace.TraceState] requirements.

    Adds [SerialModel][trulens.core.utils.serial.SerialModel] and therefore
    [pydantic.BaseModel][pydantic.BaseModel] onto the OTEL TraceState.
    """

    # Hackish: span_api.TraceState uses _dict internally.
    _dict: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)


class SpanContext(serial_utils.SerialModel, Hashable):
    """[OTEL SpanContext][opentelemetry.trace.SpanContext] requirements.

    Adds [SerialModel][trulens.core.utils.serial.SerialModel] and therefore
    [pydantic.BaseModel][pydantic.BaseModel] onto the OTEL SpanContext.

    Also adds hashing, equality, conversion, and representation methods.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # needed for enums that do not inherit from str
    )

    def __str__(self):
        return f"{self.trace_id % 0xFF:02x}/{self.span_id % 0xFF:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.trace_id + self.span_id

    def __eq__(self, other: ContextLike):
        if other is None:
            return False

        return self.trace_id == other.trace_id and self.span_id == other.span_id

    trace_id: types_schema.TraceID.PY_TYPE
    """Unique identifier for the trace.

    Each root span has a unique trace id."""

    span_id: types_schema.SpanID.PY_TYPE
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
    """Reference to the tracer that produces this SpanContext."""

    @property
    def tracer(self) -> Tracer:
        """Tracer that produced this SpanContext."""
        return self._tracer

    @staticmethod
    def of_contextlike(
        context: ContextLike, tracer: Optional[Tracer] = None
    ) -> SpanContext:
        """Convert several types that convey span/contxt identifiers into the
        common SpanContext type."""

        if isinstance(context, SpanContext):
            if tracer is not None:
                context._tracer = tracer

            return context

        if isinstance(context, span_api.SpanContext):
            # otel api SpanContext; doesn't have hashing and other things we need.
            return SpanContext(
                trace_id=context.trace_id,
                span_id=context.span_id,
                is_remote=context.is_remote,
                _tracer=tracer,
            )
        if isinstance(context, context_api.Context):
            # Context dict from OTEL.

            if len(context) == 1:
                span_encoding = next(iter(context.values()))

                return SpanContext(
                    trace_id=types_schema.TraceID.py_of_otel(
                        span_encoding.trace_id
                    ),
                    span_id=types_schema.SpanID.py_of_otel(
                        span_encoding.span_id
                    ),
                    _tracer=tracer,
                )
            else:
                raise ValueError(
                    f"Unrecognized context dict from OTEL: {context}"
                )
        if isinstance(context, dict):
            # Json encoding of SpanContext, i.e. output of
            # SpanContext.model_dump .

            context["_tracer"] = tracer
            return SpanContext.model_validate(context)

        raise ValueError(f"Unrecognized span context type: {context}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)


ContextLike = Union[
    SpanContext, span_api.SpanContext, context_api.Context, serial_utils.JSON
]
"""SpanContext types we need to deal with.

These may be the non-hashable ones coming from OTEL, the hashable ones we
defined above, or their JSON representations."""


class Span(
    pyschema_utils.WithClassInfo,
    serial_utils.SerialModel,
    trace_api.Span,
    Hashable,
):
    """[OTEL Span][opentelemetry.trace.Span] requirements.

    See also [OpenTelemetry
    Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span) and
    [OpenTelemetry Span
    specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md).

    Adds more features on top of basic OTEL API span requirements:

    - Hashable.

    - pydantic.BaseModel for validation and (de)serialization.

    - Async context manager requirements (__aenter__, __aexit__).

    - Conversions to OTEL ReadableSpan (methods starting with "otel_").
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # model_validate will fail without this
    )

    name: Optional[str] = None

    kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL

    @pydantic.field_validator("kind")
    @classmethod
    def _validate_kind(cls, v):
        return trace_api.SpanKind(v)

    context: SpanContext

    parent: Optional[SpanContext] = None

    status: trace_api.status.StatusCode = trace_api.status.StatusCode.UNSET

    @pydantic.field_validator("status")
    @classmethod
    def _validate_status(cls, v):
        return trace_api.status.StatusCode(v)

    status_description: Optional[str] = None

    events: List[
        Tuple[str, trace_api.types.Attributes, types_schema.Timestamp.PY_TYPE]
    ] = pydantic.Field(default_factory=list)
    links: trace_api._Links = pydantic.Field(default_factory=lambda: [])

    #    attributes: trace_api.types.Attributes = pydantic.Field(default_factory=dict)
    attributes: Dict = pydantic.Field(default_factory=dict)

    start_timestamp: types_schema.Timestamp.PY_TYPE = pydantic.Field(
        default_factory=types_schema.Timestamp.default_py
    )

    end_timestamp: Optional[types_schema.Timestamp.PY_TYPE] = None

    _record_exception: bool = pydantic.PrivateAttr(True)
    _set_status_on_exception: bool = pydantic.PrivateAttr(True)
    _end_on_exit: bool = pydantic.PrivateAttr(True)

    _tracer: Tracer = pydantic.PrivateAttr(None)
    """NON-STANDARD: The Tracer that produced this span."""

    @property
    def tracer(self) -> Tracer:
        """The tracer that produced this span."""
        return self._tracer

    def __hash__(self):
        return hash(self.context)

    def __init__(self, **kwargs):
        if kwargs.get("start_timestamp") is None:
            kwargs["start_timestamp"] = types_schema.Timestamp.default_py()

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

    def get_span_context(self) -> span_api.SpanContext:
        """See [OTEL get_span_context][opentelemetry.trace.span.Span.get_span_context]."""

        return self.context

    def set_status(
        self,
        status: Union[span_api.Status, span_api.StatusCode],
        description: Optional[str] = None,
    ) -> None:
        """See [OTEL set_status][opentelemetry.trace.span.Span.set_status]."""

        if isinstance(status, span_api.Status):
            if description is not None:
                raise ValueError(
                    "Ambiguous status description provided both in `status.description`"
                    " and in `description`."
                )

            assert isinstance(status.status_code, span_api.StatusCode), (
                f"Invalid status code {status.status_code} of type "
                f"{type(status.status_code)}."
            )

            self.status = span_api.StatusCode(status.status_code)
            self.status_description = status.description

        elif isinstance(status, span_api.StatusCode):
            self.status = span_api.StatusCode(status)
            self.status_description = description

        else:
            raise ValueError(f"Invalid status {status} or type {type(status)}.")

    def add_event(
        self,
        name: str,
        attributes: types_api.Attributes = None,
        timestamp: Optional[types_schema.Timestamp.OTEL_TYPE] = None,
    ) -> None:
        """See [OTEL add_event][opentelemetry.trace.span.Span.add_event].

        !!! Warning:
            As this is an OTEL requirement, we accept expected OTEL types
            instead of the ones we actually use in our classes.
        """

        self.events.append((
            name,
            attributes,
            types_schema.Timestamp.py_of_otel(timestamp)
            if timestamp is not None
            else types_schema.Timestamp.default_py(),
        ))

    def add_link(
        self,
        context: span_api.SpanContext,
        attributes: types_api.Attributes = None,
    ) -> None:
        """See [OTEL add_link][opentelemetry.trace.span.Span.add_link]."""

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def is_recording(self) -> bool:
        """See [OTEL
        is_recording][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_api.status.StatusCode.UNSET

    def set_attributes(
        self, attributes: Dict[str, types_api.AttributeValue]
    ) -> None:
        """See [OTEL
        set_attributes][opentelemetry.trace.span.Span.set_attributes]."""

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_attribute(self, key: str, value: types_api.AttributeValue) -> None:
        """See [OTEL
        set_attribute][opentelemetry.trace.span.Span.set_attribute]."""

        self.attributes[key] = value

    def record_exception(
        self,
        exception: BaseException,
        attributes: types_api.Attributes = None,
        timestamp: Optional[types_schema.Timestamp.UNION_TYPE] = None,
        escaped: bool = False,  # purpose unknown
    ) -> None:
        """See [OTEL
        record_exception][opentelemetry.trace.span.Span.record_exception].

        !!! Warning:
            As this is an OTEL requirement, we accept expected OTEL types in
            args.
        """

        if exception is None:
            raise RuntimeError("Exception must be provided.")

        # TODO: what to do here other than record the exception?
        # print(f"Encountered exception {type(exception)} in span {self}:")
        # print(exception)
        # traceback.print_exception(type(exception), exception, exception.__traceback__)

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

    def end(self, end_time: Optional[types_schema.Timestamp.UNION_TYPE] = None):
        """See [OTEL end][opentelemetry.trace.span.Span.end].

        !!! Warning:
            As this is an OTEL requirement, we accept expected OTEL types in
            args.
        """

        if end_time is None:
            self.end_timestamp = types_schema.Timestamp.default_py()
        else:
            self.end_timestamp = types_schema.Timestamp.py(end_time)

        if self.is_recording():
            self.set_status(
                trace_api.status.Status(trace_api.status.StatusCode.OK)
            )

    # context manager requirement
    def __enter__(self) -> Span:
        """See [OTEL __enter__][opentelemetry.trace.span.Span.__enter__]."""

        # Span can be used as a context manager to automatically handle ending
        # and exception recording.

        return self

    # context manager requirement
    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """See [OTEL __exit__][opentelemetry.trace.span.Span.__exit__]."""

        if exc_val is not None:
            self.record_exception(exception=exc_val)

        if self._end_on_exit:
            self.end()

        return False  # don't suppress exceptions

    # async context manager requirement
    async def __aenter__(self) -> Span:
        return self.__enter__()

    # async context manager requirement
    async def __aexit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc_val, exc_tb)

    # Rest of these methods are for exporting spans to ReadableSpan. All are not
    # standard OTEL but values for OTEL ReadableSpan.

    @staticmethod
    def otel_context_of_context(context: SpanContext) -> trace_api.SpanContext:
        return trace_api.SpanContext(
            trace_id=types_schema.TraceID.otel_of_py(context.trace_id),
            span_id=types_schema.SpanID.otel_of_py(context.span_id),
            is_remote=False,
        )

    def otel_context(self) -> types_api.SpanContext:
        return self.otel_context_of_context(self.context)

    def otel_parent_context(self) -> Optional[types_api.SpanContext]:
        if self.parent is None:
            return None

        return self.otel_context_of_context(self.parent)

    def otel_attributes(self) -> types_api.Attributes:
        return types_schema.flatten_lensed_attributes(self.attributes)

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
        return [
            (a, b, types_schema.Timestamp.otel_of_py(c))
            for (a, b, c) in self.events
        ]

    def otel_links(self) -> List[types_api.Link]:
        return self.links

    def otel_start_timestamp(self) -> types_schema.Timestamp.OTEL_TYPE:
        return types_schema.Timestamp.otel_of_py(self.start_timestamp)

    def otel_end_timestamp(self) -> Optional[types_schema.Timestamp.OTEL_TYPE]:
        if self.end_timestamp is None:
            return None
        return types_schema.Timestamp.otel_of_py(self.end_timestamp)

    def was_exported_to(
        self, to: Hashable, mark_exported: bool = False
    ) -> bool:
        ret = to in self.exported_to
        if mark_exported:
            self.exported_to.add(to)
        return ret

    def otel_freeze(
        self,
    ) -> trace_sdk.ReadableSpan:
        """Convert span to an OTEL compatible span for exporting to OTEL collectors."""

        return trace_sdk.ReadableSpan(
            name=self.name,
            context=self.otel_context(),
            parent=self.otel_parent_context(),
            resource=self.otel_resource(),
            attributes=self.otel_attributes(),
            events=self.otel_events(),
            links=self.otel_links(),
            kind=self.otel_kind(),
            instrumentation_info=None,  # TODO(SNOW-1711959)
            status=self.otel_status(),
            start_time=self.otel_start_timestamp(),
            end_time=self.otel_end_timestamp(),
            instrumentation_scope=None,  # TODO(SNOW-1711959)
        )


def _default_context_factory(
    name: str,
) -> Callable[[], contextvars.ContextVar[SpanContext]]:
    """Create a default span context contextvar factory.

    Includes the given name in the contextvar name. The default context is a
    non-recording context.
    """

    def create():
        return contextvars.ContextVar(
            f"context_{name}_{python_utils.context_id()}",
            default=SpanContext(
                trace_id=types_schema.TraceID.INVALID_OTEL,
                span_id=types_schema.SpanID.INVALID_OTEL,
            ),
        )

    return create


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

    def __str__(self):
        return (
            type(self).__name__
            + " "
            + (self._instrumenting_module_name or "")
            + " "
            + (self._instrumenting_library_version or "")
        )

    def __repr__(self):
        return str(self)

    def __init__(self, _span_context_cvar: SpanContext, **kwargs):
        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)

        self._span_context_cvar = _span_context_cvar

    _span_context_cvar: contextvars.ContextVar[SpanContext] = (
        pydantic.PrivateAttr(default_factory=_default_context_factory("Tracer"))
    )

    @property
    def current_span_context(self) -> SpanContext:
        return self._span_context_cvar.get()

    def current_span_id(self) -> types_schema.SpanID.PY_TYPE:
        return self.current_span_context.span_id

    def current_trace_id(self) -> types_schema.TraceID.PY_TYPE:
        return self.current_span_context.trace_id

    def start_span(
        self,
        name: Optional[str] = None,
        context: Optional[ContextLike] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.types.Attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[types_schema.Timestamp.UNION_TYPE] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        cls: Optional[Type[Span]] = None,  # non-standard
        end_on_exit: bool = True,  # non-standard
        **kwargs,  # non-standard
    ) -> Span:
        """See [OTEL
        Tracer.start_span][opentelemetry.trace.Tracer.start_span].

        !!! Warning:
            As this is an OTEL requirement, we accept expected OTEL types in
            args.

        Args:
            cls: Class of span to create. Defaults to the class set in the
            tracer.

            trace_id: Trace id to use. Defaults to the current trace id.

            *args: Additional arguments to pass to the span.

            **kwargs: Additional keyword arguments to pass to the span.
        """

        if (
            context is None
            or (parent_context := SpanContext.of_spancontextlike(context))
            is None
        ):
            parent_context = self.current_span_context

        new_context = SpanContext(
            trace_id=parent_context.trace_id,
            span_id=types_schema.SpanID.rand_otel(),
            _tracer=self,
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
            start_timestamp=types_schema.Timestamp.py(start_time)
            if start_time
            else None,
            _record_exception=record_exception,
            _set_status_on_exception=set_status_on_exception,
            _end_on_exit=end_on_exit,
            _tracer=self,
            **kwargs,
        )

        return new_span

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: Optional[str] = None,
        context: Optional[ContextLike] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[types_schema.Timestamp.UNION_TYPE] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        cls: Optional[Type[Span]] = None,  # non-standard
        **kwargs,  # non-standard
    ):
        """See [OTEL
        Tracer.start_as_current_span][opentelemetry.trace.Tracer.start_as_current_span].

        !!! Warning:
            As this is an OTEL requirement, we accept expected OTEL types in
            args.
        """

        # TODO: Make this agnostic context manager to match OTEL.
        # TODO: Make this usable as a decorator to match OTEL.

        # Create a new span. Context controls its ending and recording of exception.
        with self.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            cls=cls,
            end_on_exit=end_on_exit,
            **kwargs,
        ) as span:
            # Set current span context to that of the new span.
            with python_utils.with_context({
                self._span_context_cvar: span.context
            }):
                try:
                    yield span
                finally:
                    pass

    @contextlib.asynccontextmanager
    async def astart_as_current_span(
        self,
        name: Optional[str] = None,
        context: Optional[trace_api.context.Context] = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: trace_api.attributes = None,
        links: trace_api._Links = None,
        start_time: Optional[types_schema.Timestamp.UNION_TYPE] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        cls: Optional[Type[Span]] = None,  # non-standard
        **kwargs,  # non-standard
    ):
        """Not otel standard but mimics the sync version.

        In OTEL, `start_as_current_span` works both for sync and async.
        """

        # Create a new span.
        async with self.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            cls=cls,
            end_on_exit=end_on_exit,
            **kwargs,
        ) as span:
            # Set current span context to that of the new span.
            async with python_utils.awith_context({
                self._span_context_cvar: span.context
            }):
                try:
                    yield span
                finally:
                    pass


class TracerProvider(serial_utils.SerialModel, trace_api.TracerProvider):
    """[OTEL TracerProvider][opentelemetry.trace.TracerProvider]
    requirements."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(Tracer)
    """NON-STANDARD: The default tracer class to use when creating tracers."""

    _span_context_cvar: contextvars.ContextVar[SpanContext] = (
        pydantic.PrivateAttr(
            default_factory=_default_context_factory("TracerProvider")
        )
    )

    @property
    def current_span_context(self) -> SpanContext:
        """NON-STANDARD: The current span context."""

        return self._span_context_cvar.get()

    @property
    def current_trace_id(self) -> types_schema.TraceID.PY_TYPE:
        """NON-STANDARD: The current trace id."""

        return self.current_span_context.trace_id

    @property
    def current_span_id(self) -> types_schema.SpanID.PY_TYPE:
        """NON-STANDARD: The current span id."""

        return self.current_span_context.span_id

    def __init__(self):
        super().__init__()

        self._span_context_cvar.set(
            SpanContext(
                span_id=types_schema.SpanID.rand_otel(),
                trace_id=types_schema.TraceID.rand_otel(),
            )
        )

        print(
            f"{text_utils.UNICODE_SQUID} TruLens root context={self._span_context_cvar.get()}"
        )

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
            _span_context_cvar=self._span_context_cvar,
        )

        return tracer
