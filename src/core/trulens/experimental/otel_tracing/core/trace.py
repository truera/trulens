"""Implementation of recording that resembles the tracing process in OpenTelemetry.

!!! Note
    Most of the module is (EXPERIMENTAL: otel-tracing) though it includes some existing
    non-experimental classes moved here to resolve some circular import issues.

This module is likely temporary and will be replaced by actual OpenTelemetry sdk
components or implementations that are compatible with its API.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import os
import random
import threading as th
from threading import Lock
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import uuid

from opentelemetry import context as ot_context
from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import opentelemetry.trace as ot_trace
from opentelemetry.trace import status as trace_status
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing.core import otel as mod_otel
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils

logger = logging.getLogger(__name__)

INSTRUMENT: str = "__tru_instrumented"
"""Attribute name to be used to flag instrumented objects/methods/others."""

APPS: str = "__tru_apps"
"""Attribute name for storing apps that expect to be notified of calls."""


TTimestamp = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NUM_TIMESTAMP_BITS = 64


def lens_of_flat_key(key: str) -> serial_utils.Lens:
    """Convert a flat dict key to a lens."""
    lens = serial_utils.Lens()
    for step in key.split("."):
        lens = lens[step]

    return lens


class TraceState(serial_utils.SerialModel, ot_span.TraceState):
    # Hackish: ot_span.TraceState uses _dict internally.
    _dict: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)


class SpanContext(
    serial_utils.SerialModel
):  # should be compatible with ot_span.SpanContext
    """Identifiers for a span."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def new_trace_id():
        return int(random.getrandbits(128) & ot_span._TRACE_ID_MAX_VALUE)

    # otel requirement
    trace_id: int = pydantic.Field(default_factory=new_trace_id)
    """Unique identifier for the trace.

    Each root span has a unique trace id."""

    # otel requirement
    span_id: int = pydantic.Field(
        default_factory=lambda: int(
            random.getrandbits(64) & ot_span._SPAN_ID_MAX_VALUE
        )
    )
    """Identifier for the span.

    Meant to be at least unique within the same trace_id.
    """

    # otel requirement
    trace_flags: ot_trace.TraceFlags = ot_trace.TraceFlags(0)

    # otel requirement
    trace_state: TraceState = pydantic.Field(default_factory=TraceState)

    # otel requirement
    is_remote: bool = False

    tracer: Tracer = pydantic.Field(exclude=True)
    """Reference to the tracer that created this span."""

    def __str__(self):
        return f"{self.trace_id % 0xFF:02x}/{self.span_id % 0xFF:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.trace_id + self.span_id

    def __eq__(self, other):
        if other is None:
            return False
        return self.trace_id == other.trace_id and self.span_id == other.span_id


ContextLike = Union[SpanContext, ot_span.SpanContext]


class Span(
    serial_utils.SerialModel, ot_span.Span
):  # ot_span.Span is mostly abstract
    """An OTEL-compatible span.

    See [Span][opentelemetry.trace.Span].

    See also [OpenTelemetry
    Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span) and
    [OpenTelemetry Span
    specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # for opentelemetry.trace.Span requirement
    name: str = "unnamed"

    def __str__(self):
        return (
            f"{type(self).__name__}({self.name}, {self.context}->{self.parent})"
        )

    def __repr__(self):
        return str(self)

    def update_name(self, name: str) -> None:
        """See [update_name][opentelemetry.trace.span.Span.update_name]."""

        self.name = name

    # for opentelemetry.trace.Span requirement
    kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL
    """Kind of span."""

    # for opentelemetry.trace.Span requirement
    status: trace_status.StatusCode = trace_status.StatusCode.UNSET
    """Status of the span as per OpenTelemetry Span requirements."""

    def set_status(
        self,
        status: Union[ot_span.Status, ot_span.StatusCode],
        description: Optional[str] = None,
    ) -> None:
        """See [set_status][opentelemetry.trace.span.Span.set_status]."""

        if isinstance(status, ot_span.Status):
            if description is not None:
                raise ValueError(
                    "Ambiguous status description provided both in `status.description` and in `description`."
                )

            self.status = status.status_code
            self.status_description = status.description
        else:
            self.status = status
            self.status_description = description

    # for opentelemetry.trace.Span requirement
    status_description: Optional[str] = None
    """Status description as per OpenTelemetry Span requirements."""

    # for opentelemetry.trace.Span requirement
    events: List[Tuple[str, ot_types.Attributes, TTimestamp]] = pydantic.Field(
        default_factory=list
    )
    """Events recorded in the span.

    !!! Warning

        Events in OpenTelemetry seem to be synonymous to logs. Do not store
        anything we want to query or process in events.
    """

    def add_event(
        self,
        name: str,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
    ) -> None:
        """See [add_event][opentelemetry.trace.span.Span.add_event]."""

        self.events.append((name, attributes, timestamp or time.time_ns()))

    # for opentelemetry.trace.Span requirement
    links: Dict[SpanContext, ot_types.Attributes] = pydantic.Field(
        default_factory=dict
    )
    """Relationships to other spans with attributes on each link."""

    def add_link(
        self,
        context: ot_span.SpanContext,
        attributes: ot_types.Attributes = None,
    ) -> None:
        """See [add_link][opentelemetry.trace.span.Span.add_link]."""

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def is_recording(self) -> bool:
        """See [is_recording][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_status.StatusCode.UNSET

    # for opentelemetry.trace.Span requirement
    _attributes: Dict[str, Any] = pydantic.PrivateAttr(
        default_factory=serial_utils.LensedDict
    )

    @pydantic.computed_field
    @property
    def attributes(self) -> Dict[str, ot_types.AttributeValue]:
        return self._attributes

    def set_attributes(
        self, attributes: Dict[str, ot_types.AttributeValue]
    ) -> None:
        """See [set_attributes][opentelemetry.trace.span.Span.set_attributes]."""

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_attribute(self, key: str, value: ot_types.AttributeValue) -> None:
        """See [set_attribute][opentelemetry.trace.span.Span.set_attribute]."""

        self.attributes[lens_of_flat_key(key)] = value

    # for opentelemetry.trace.Span requirement
    context: SpanContext = pydantic.Field()
    """Identifiers."""

    def get_span_context(self) -> ot_span.SpanContext:
        """See [get_span_context][opentelemetry.trace.span.Span.get_span_context]."""

        return self.context

    # for opentelemetry.trace.Span requirement
    parent: Optional[SpanContext] = pydantic.Field(None)
    """Optional parent identifier."""

    live_parent_span: Optional[Span] = pydantic.Field(None, exclude=True)
    live_children_spans: List[Span] = pydantic.Field(
        default_factory=list, exclude=True
    )

    def record_exception(
        self,
        exception: Exception,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
        escaped: bool = False,
    ) -> None:
        """See [record_exception][opentelemetry.trace.span.Span.record_exception]."""

        self.status = trace_status.StatusCode.ERROR

        self.add_event(self.vendor_attr("exception"), attributes, timestamp)

    error: Optional[Exception] = pydantic.Field(None)
    """Optional error if the observed computation raised an exception."""

    start_timestamp: Optional[int] = pydantic.Field(
        default_factory=time.time_ns
    )
    """Start time in nanoseconds since epoch."""

    end_timestamp: Optional[int] = pydantic.Field(None)
    """End time in nanoseconds since epoch.

    None if not yet finished."""

    def end(self, end_time: Optional[TTimestamp] = None):
        """See [end][opentelemetry.trace.span.Span.end]"""

        if end_time:
            self.end_timestamp = end_time
        else:
            self.end_timestamp = time.time_ns()

        self.status = trace_status.StatusCode.OK

    @property
    def tracer(self):
        return self.context.tracer

    def finish(self):
        self.end_timestamp = time.time_ns()

    async def afinish(self):
        self.finish()

    def iter_children(
        self, transitive: bool = True, include_phantom: bool = False
    ) -> Iterable[Span]:
        """Iterate over all spans that are children of this span.

        Args:
            transitive: Iterate recursively over children.

            include_phantom: Include phantom spans. If not set, phantom spans
                will not be included but will be iterated over even if
                transitive is false.
        """

        for child_span in self.live_children_spans:
            if isinstance(child_span, PhantomSpan) and not include_phantom:
                # Note that transitive being false is ignored if phantom is skipped.
                yield from child_span.iter_children(
                    transitive=transitive, include_phantom=include_phantom
                )
            else:
                yield child_span
                if transitive:
                    yield from child_span.iter_children(
                        transitive=transitive,
                        include_phantom=include_phantom,
                    )

    def iter_family(self, include_phantom: bool = False) -> Iterable[Span]:
        """Iterate itself and all children transitively."""

        if (not isinstance(self, PhantomSpan)) or include_phantom:
            yield self

        yield from self.iter_children(
            include_phantom=include_phantom, transitive=True
        )

    def total_cost(self) -> base_schema.Cost:
        """Total costs of this span and all its transitive children."""

        total = base_schema.Cost()

        for span in self.iter_family(include_phantom=True):
            if isinstance(span, WithCost) and span.cost is not None:
                total += span.cost

        return total


class OTELExportable(Span):
    """Methods for converting a span to an OTEL span."""

    @staticmethod
    def otel_context_of_context(context: SpanContext) -> ot_span.SpanContext:
        return ot_span.SpanContext(
            trace_id=context.trace_id,
            span_id=context.span_id,
            is_remote=False,
        )

    def otel_name(self) -> str:
        return self.name

    def otel_context(self) -> ot_types.SpanContext:
        return self.otel_context_of_context(self.context)

    def otel_parent_context(self) -> Optional[ot_types.SpanContext]:
        if self.parent is None:
            return None
        return self.otel_context_of_context(self.parent)

    def otel_attributes(self) -> ot_types.Attributes:
        return mod_otel.flatten_lensed_attributes(self.attributes)

    def otel_kind(self) -> ot_types.SpanKind:
        return ot_trace.SpanKind.INTERNAL

    def otel_status(self) -> trace_status.Status:
        if self.error is not None:
            return trace_status.Status(
                status_code=trace_status.StatusCode.ERROR,
                description=str(self.error),
            )

        return trace_status.Status(status_code=trace_status.StatusCode.OK)

    def otel_resource_attributes(self) -> Dict[str, Any]:
        return {
            "service.namespace": "trulens",
        }

    def otel_resource(self) -> Resource:
        return Resource(attributes=self.otel_resource_attributes())

    def otel_freeze(self) -> otsdk_trace.ReadableSpan:
        """Convert span to an OTEL compatible span for exporting to OTEL collectors.

        !!! Warning
            This is an experimental feature. OTEL integration is ongoing.
        """

        return otsdk_trace.ReadableSpan(
            name=self.otel_name(),
            context=self.otel_context(),
            parent=self.otel_parent_context(),
            resource=self.otel_resource(),
            attributes=self.otel_attributes(),
            events=[],
            links=[],
            kind=self.otel_kind(),
            instrumentation_info=None,
            status=self.otel_status(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            instrumentation_scope=None,
        )


class PhantomSpan(Span):
    """A span type that indicates that it does not correspond to a
    computation to be recorded but instead is an element of the tracing system.

    It is to be removed from the spans presented to the users.
    """


class LiveSpan(Span):
    """A a span type that indicates that it contains live python objects.

    It is to be converted to a non-live span before being output to the user or
    otherwise.
    """


class PhantomSpanRecordingContext(PhantomSpan, OTELExportable):
    """Tracks the context of an app used as a context manager."""

    recording: Optional[Any] = pydantic.Field(None, exclude=True)
    # TODO: app.RecordingContext # circular import issues

    def otel_resource_attributes(self) -> Dict[str, Any]:
        ret = super().otel_resource_attributes()

        ret[ResourceAttributes.SERVICE_NAME] = (
            self.recording.app.app_name if self.recording is not None else None
        )

        return ret

    def finish(self):
        super().finish()

        self._finalize_recording()

    async def afinish(self):
        await super().afinish()

        self._finalize_recording()

    def _finalize_recording(self):
        assert self.recording is not None

        app = self.recording.app

        for span in Tracer.find_each_child(
            span=self, span_filter=lambda s: isinstance(s, LiveSpanCall)
        ):
            app.on_new_root_span(recording=self.recording, root_span=span)

        app.on_new_recording_span(recording_span=self)

    def otel_name(self) -> str:
        return "trulens.recording"


class SpanCall(OTELExportable):
    """Non-live fields of a function call span."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    call_id: Optional[uuid.UUID] = pydantic.Field(None)
    """Unique identifier for the call."""

    stack: Optional[List[mod_record_schema.RecordAppCallMethod]] = (
        pydantic.Field(None)
    )
    """Call stack of instrumented methods only."""

    sig: Optional[inspect.Signature] = pydantic.Field(None)
    """Signature of the function."""

    func_name: Optional[str] = None
    """Function name."""

    pid: Optional[int] = None
    """Process id."""

    tid: Optional[int] = None
    """Thread id."""

    def finish(self):
        super().finish()

        self.set_attribute(ResourceAttributes.PROCESS_PID, self.pid)
        self.set_attribute("thread.id", self.tid)  # TODO: semconv

        self.set_attribute("trulens.call_id", str(self.call_id))
        self.set_attribute("trulens.stack", json_utils.jsonify(self.stack))
        self.set_attribute("trulens.sig", str(self.sig))

    def otel_name(self) -> str:
        return f"trulens.call.{self.func_name}"


class LiveSpanCall(LiveSpan, SpanCall):
    """Track a function call."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # call: Optional[mod_record_schema.RecordAppCall] = None
    live_obj: Optional[Any] = pydantic.Field(None, exclude=True)
    """Self object if method call."""

    live_cls: Optional[Type] = pydantic.Field(None, exclude=True)
    """Class if method/static/class method call."""

    live_func: Optional[Callable] = pydantic.Field(None, exclude=True)
    """Function object."""

    live_args: Optional[Tuple[Any, ...]] = pydantic.Field(None, exclude=True)
    """Positional arguments to the function call."""

    live_kwargs: Optional[Dict[str, Any]] = pydantic.Field(None, exclude=True)
    """Keyword arguments to the function call."""

    live_bindings: Optional[inspect.BoundArguments] = pydantic.Field(
        None, exclude=True
    )
    """Bound arguments to the function call if can be bound."""

    live_ret: Optional[Any] = pydantic.Field(None, exclude=True)
    """Return value of the function call.

    Exclusive with `error`.
    """

    live_error: Optional[Any] = pydantic.Field(None, exclude=True)
    """Error raised by the function call.

    Exclusive with `ret`.
    """

    def finish(self):
        super().finish()

        if self.live_cls is not None:
            self.set_attribute(
                "trulens.cls",
                pyschema_utils.Class.of_class(self.live_cls).model_dump(),
            )

        if self.live_func is not None:
            self.set_attribute(
                "trulens.func",
                pyschema_utils.FunctionOrMethod.of_callable(
                    self.live_func
                ).model_dump(),
            )

        """
        # Redundant with bindings
        if self.live_args is not None:
            self.set_attribute("trulens.args", json_utils.jsonify(self.live_args))
        if self.live_kwargs is not None:
            self.set_attribute("trulens.kwargs", json_utils.jsonify(self.live_kwargs))
        """

        if self.live_ret is not None:
            self.set_attribute("trulens.ret", json_utils.jsonify(self.live_ret))

        if self.live_bindings is not None:
            self.set_attribute(
                "trulens.bindings",
                pyschema_utils.Bindings.of_bound_arguments(
                    self.live_bindings, arguments_only=True, skip_self=True
                ).model_dump()["kwargs"],
            )

        if self.live_error is not None:
            self.set_attribute(
                "trulens.error", json_utils.jsonify(self.live_error)
            )


class WithCost(LiveSpan):
    """Mixin to indicate the span has costs tracked."""

    cost: base_schema.Cost = pydantic.Field(default_factory=base_schema.Cost)
    """Cost of the computation spanned."""

    endpoint: Optional[Any] = pydantic.Field(None, exclude=True)
    """Endpoint handling cost extraction for this span/call."""

    def finish(self):
        super().finish()

        self.set_attribute("trulens.cost", self.cost.model_dump())

    def __init__(self, cost: Optional[base_schema.Cost] = None, **kwargs):
        if cost is None:
            cost = base_schema.Cost()

        super().__init__(cost=cost, **kwargs)


class LiveSpanCallWithCost(LiveSpanCall, WithCost):
    pass


class Tracer(pydantic.BaseModel, ot_trace.Tracer):
    """OTEL-compatible Tracer.

    See [Tracer][opentelemetry.trace.Tracer].
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    tracer_provider: TracerProvider

    instrumenting_module_name: Optional[str] = None
    """Name of the library/module that is instrumenting the code."""

    instrumenting_library_version: Optional[str] = None
    """Version of the library that is instrumenting the code."""

    def __init__(self, tracer_provider: TracerProvider, **kwargs):
        super().__init__(tracer_provider=tracer_provider, **kwargs)

    def __str__(self):
        return f"{type(self).__name__} {self.instrumenting_module_name} {self.instrumenting_library_version}"

    def __repr__(self):
        return str(self)

    @property
    def context(self):
        return self.tracer_provider.context

    @property
    def trace_id(self):
        return self.tracer_provider.trace_id

    @property
    def spans(self):
        return self.tracer_provider.spans

    # opentelemetry.trace.Tracer requirement
    def start_span(
        self,
        name: Optional[str] = None,
        *,
        context: Optional[ot_context.context.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_trace.types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        cls: Optional[Type[Span]] = OTELExportable,  # non-standard attribute
        **kwargs,  # non-standard attributes
    ) -> Span:
        if context is None:
            parent_context = self.context.get()
        else:
            if len(context) != 1:
                raise ValueError("Only one context span is allowed.")
            parent_span_encoding = next(iter(context.values()))

            parent_context = SpanContext(
                trace_id=parent_span_encoding.trace_id,
                span_id=parent_span_encoding.span_id,
                tracer=self,
            )

        new_context = SpanContext(trace_id=self.trace_id, tracer=self)

        if cls is None:
            cls = OTELExportable

        if name is None:
            name = python_utils.class_name(cls)

        new_span = cls(
            name=name,
            context=new_context,
            tracer=self,
            parent=parent_context,
            **kwargs,
        )

        self.spans[new_context] = new_span

        if parent_context in self.spans:
            parent_span = self.spans[parent_context]
            parent_span.live_children_spans.append(new_span)
            new_span.live_parent_span = parent_span

        return new_span

    # opentelemetry.trace.Tracer requirement
    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: Optional[str] = None,
        *,
        context: Optional[ot_context.context.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_types.Attributes = None,
        links: ot_trace._Links = None,
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

        token = self.context.set(span.context)

        yield span

        self.context.reset(token)
        span.finish()

    @staticmethod
    def _fill_stacks(
        span: Span,
        get_method_path: Callable,
        stack: List[mod_record_schema.RecordAppCallMethod] = [],
    ):
        if isinstance(span, LiveSpanCall):
            path = get_method_path(obj=span.live_obj, func=span.live_func)

            frame_ident = mod_record_schema.RecordAppCallMethod(
                path=path
                if path is not None
                else serial_utils.Lens().static,  # placeholder path for functions
                method=pyschema_utils.Method.of_method(
                    span.live_func, obj=span.live_obj, cls=span.live_cls
                ),
            )

            stack = stack + [frame_ident]
            span.stack = stack

        for subspan in span.iter_children(transitive=False):
            Tracer._fill_stacks(
                subspan, stack=stack, get_method_path=get_method_path
            )

    def _call_of_spancall(
        self, span: LiveSpanCall
    ) -> mod_record_schema.RecordAppCall:
        """Convert a SpanCall to a RecordAppCall."""

        args = (
            dict(span.live_bindings.arguments)
            if span.live_bindings is not None
            else None
        )
        if args is not None:
            if "self" in args:
                del args["self"]  # remove self

        assert span.start_timestamp is not None
        assert span.end_timestamp is not None

        return mod_record_schema.RecordAppCall(
            call_id=str(span.call_id),
            stack=span.stack,
            args=args,
            rets=json_utils.jsonify(span.live_ret),
            error=str(span.live_error),
            perf=base_schema.Perf.of_ns_timestamps(
                start_ns_timestamp=span.start_timestamp,
                end_ns_timestamp=span.end_timestamp,
            ),
            pid=span.pid,
            tid=span.tid,
        )

    def record_of_root_span(
        self, recording: Any, root_span: LiveSpanCall
    ) -> mod_record_schema.Record:
        """Convert a root span to a record.

        This span has to be a call span so we can extract things like main input and output.
        """

        assert isinstance(root_span, LiveSpanCall), type(root_span)

        app = recording.app

        self._fill_stacks(root_span, get_method_path=app.get_method_path)

        root_perf = (
            base_schema.Perf.of_ns_timestamps(
                start_ns_timestamp=root_span.start_timestamp,
                end_ns_timestamp=root_span.end_timestamp,
            )
            if root_span.end_timestamp is not None
            else None
        )

        if isinstance(root_span, WithCost):
            root_cost = root_span.total_cost()
        else:
            root_cost = base_schema.Cost()

        calls = []
        if isinstance(root_span, LiveSpanCall):
            calls.append(self._call_of_spancall(root_span))

        spans = [root_span]

        for span in root_span.iter_children(include_phantom=True):
            if isinstance(span, LiveSpanCall):
                calls.append(self._call_of_spancall(span))

            spans.append(span)

        bindings = root_span.live_bindings
        main_error = root_span.live_error

        if bindings is not None:
            main_input = app.main_input(
                func=root_span.live_func,
                sig=root_span.sig,
                bindings=root_span.live_bindings,
            )
            if main_error is None:
                main_output = app.main_output(
                    func=root_span.live_func,
                    sig=root_span.sig,
                    bindings=root_span.live_bindings,
                    ret=root_span.live_ret,
                )
            else:
                main_output = None
        else:
            main_input = None
            main_output = None

        record = mod_record_schema.Record(
            record_id="placeholder",
            app_id=app.app_id,
            main_input=json_utils.jsonify(main_input),
            main_output=json_utils.jsonify(main_output),
            main_error=json_utils.jsonify(main_error),
            calls=calls,
            perf=root_perf,
            cost=root_cost,
            experimental_otel_spans=spans,
        )

        # record_id determinism
        record.record_id = json_utils.obj_id_of_obj(record, prefix="record")

        return record

    @staticmethod
    def find_each_child(span: Span, span_filter: Callable) -> Iterable[Span]:
        """For each family rooted at each child of this span, find the top-most
        span that satisfies the filter."""

        for child_span in span.live_children_spans:
            if span_filter(child_span):
                yield child_span
            else:
                yield from Tracer.find_each_child(child_span, span_filter)

    def records_of_recording(
        self, recording: PhantomSpanRecordingContext
    ) -> Iterable[mod_record_schema.Record]:
        """Convert a recording based on spans to a list of records."""

        for root_span in Tracer.find_each_child(
            span=recording, span_filter=lambda s: isinstance(s, LiveSpanCall)
        ):
            assert isinstance(root_span, LiveSpanCall)
            yield self.record_of_root_span(
                recording=recording, root_span=root_span
            )

    @contextlib.contextmanager
    def _span(self, cls, **kwargs):
        span = self.start_span(cls=cls, **kwargs)

        token = self.context.set(span.context)

        try:
            yield span
        except BaseException as e:
            span.error = e
        finally:
            self.context.reset(token)
            span.finish()
            if span.error is None:
                span.status = ot_trace.status.StatusCode.OK
            else:
                span.status = ot_trace.status.StatusCode.ERROR
                span.status_description = str(span.error)
                raise span.error

    @contextlib.asynccontextmanager
    async def _aspan(self, cls, **kwargs):
        span = self.start_span(cls=cls, **kwargs)

        token = self.context.set(span.context)

        try:
            yield span
        except BaseException as e:
            span.error = e
        finally:
            self.context.reset(token)
            await span.afinish()
            if span.error is None:
                span.status = ot_trace.status.StatusCode.OK
            else:
                span.status = ot_trace.status.StatusCode.ERROR
                span.status_description = str(span.error)
                raise span.error

    # context manager
    def recording(self):
        return self._span(
            name="trulens.recording", cls=PhantomSpanRecordingContext
        )

    # context manager
    def method(self, method_name: str):
        return self._span(name="trulens.call." + method_name, cls=LiveSpanCall)

    # context manager
    def cost(self, method_name: str, cost: Optional[base_schema.Cost] = None):
        return self._span(
            name="trulens.call." + method_name,
            cls=LiveSpanCallWithCost,
            cost=cost,
        )

    # context manager
    def phantom(self):
        return self._span(name="trulens.phantom", cls=PhantomSpan)

    # context manager
    async def arecording(self):
        return self._aspan(
            name="trulens.recording", cls=PhantomSpanRecordingContext
        )

    # context manager
    async def amethod(self, method_name: str):
        return self._aspan(name="trulens.call." + method_name, cls=LiveSpanCall)

    # context manager
    async def acost(
        self, method_name: str, cost: Optional[base_schema.Cost] = None
    ):
        return self._aspan(
            name="trulens.call." + method_name,
            cls=LiveSpanCallWithCost,
            cost=cost,
        )

    # context manager
    async def aphantom(self):
        return self._aspan(name="trulens.phantom", cls=PhantomSpan)


class NullTracer(Tracer):
    """Tracer that does not save the spans it makes."""

    @contextlib.contextmanager
    def _span(self, cls, **kwargs):
        # TODO: this adds span to global list, don't do this
        span = self.start_span(cls=cls, **kwargs)

        token = self.context.set(span.context)

        error = None

        try:
            yield span
        except BaseException as e:
            # ignore exception since spans are also ignored/not recorded
            error = e
        finally:
            self.context.reset(token)
            if error is not None:
                raise error

    @contextlib.asynccontextmanager
    async def _aspan(self, cls, **kwargs):
        span = self.start_span(cls=cls, **kwargs)
        token = self.context.set(span.context)

        error = None

        try:
            yield span
        except BaseException as e:
            # ignore exception since spans are also ignored/not recorded
            error = e
        finally:
            self.context.reset(token)
            if error is not None:
                raise error

    @property
    def spans(self):
        return []


class TracerProvider(
    pydantic.BaseModel, ot_trace.TracerProvider, python_utils.Singleton
):
    """OTEL-compatible TracerProvider.

    See [TracerProvider][opentelemetry.trace.TracerProvider].
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    spans: Dict[SpanContext, Span] = pydantic.Field(
        default_factory=dict, exclude=True
    )
    """Spans are shared across all Tracers."""

    tracers: Dict[str, Tracer] = pydantic.Field(
        default_factory=dict, exclude=True
    )

    _context: contextvars.ContextVar[SpanContext] = pydantic.PrivateAttr(
        default_factory=lambda: contextvars.ContextVar("context", default=None)
    )

    trace_id: int = pydantic.Field(
        default_factory=SpanContext.new_trace_id, exclude=True
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    @property
    def context(self):
        return self._context

    @contextlib.contextmanager
    def trace(self):
        with trulens_tracer().recording() as root:
            tok = self.context.set(root.context)
            yield root

        self.context.reset(tok)

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[ot_types.Attributes] = None,
    ):
        if instrumenting_module_name in self.tracers:
            return self.tracers[instrumenting_module_name]

        temp = Tracer(
            instrumenting_module_name=instrumenting_module_name,
            instrumenting_library_version=instrumenting_library_version,
            tracer_provider=self,
        )

        self.tracers[instrumenting_module_name] = temp
        return temp


tracer_provider = TracerProvider()
"""Global tracer provider.
All traces are mady by this provider.
"""


@functools.cache
def trulens_tracer():
    from trulens.core import __version__

    return tracer_provider.get_tracer(
        "trulens.experimental.otel_tracing.core.trace", __version__
    )


T = TypeVar("T")


class TracingCallbacks(wrap_utils.CallableCallbacks[T]):
    """Extension of CallableCallbacks that adds tracing to the wrapped callable
    as implemented using tracer and spans."""

    @classmethod
    def on_callable_wrapped(
        cls, wrapper: Callable[..., Any], **kwargs: Dict[str, Any]
    ):
        return super().on_callable_wrapped(wrapper=wrapper, **kwargs)

    def __init__(
        self,
        func_name: str,
        span_type: Type[LiveSpanCall] = LiveSpanCall,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(**kwargs)

        self.func_name: str = func_name

        self.obj: Optional[object] = None
        self.obj_cls: Optional[Type] = None
        self.obj_id: Optional[int] = None

        # tracer = get_tracer()

        if not issubclass(span_type, LiveSpanCall):
            raise ValueError("span_type must be a subclass of LiveSpanCall.")

        self.span_context = trulens_tracer()._span(
            span_type, name="trulens.call." + func_name
        )
        self.span = self.span_context.__enter__()

    def on_callable_call(
        self, bindings: inspect.BoundArguments, **kwargs: Dict[str, Any]
    ) -> inspect.BoundArguments:
        temp = super().on_callable_call(bindings=bindings, **kwargs)

        if "self" in bindings.arguments:
            self.obj = bindings.arguments["self"]
            self.obj_cls = type(self.obj)
            self.obj_id = id(self.obj)

        span = self.span
        span.pid = os.getpid()
        span.tid = th.get_native_id()

        span.start_timestamp = time.time_ns()

        return temp

    def on_callable_end(self):
        span = self.span

        # SpanCall attributes
        span.call_id = self.call_id
        span.func_name = self.func_name
        span.sig = self.sig

        span.end_timestamp = time.time_ns()

        # LiveSpanCall attributes
        span.live_obj = self.obj
        span.live_cls = self.obj_cls
        span.live_func = self.func
        span.live_args = self.call_args
        span.live_kwargs = self.call_kwargs
        span.live_bindings = self.bindings
        span.live_ret = self.ret
        span.live_error = self.error

        if self.error is not None:
            self.span_context.__exit__(
                type(self.error), self.error, self.error.__traceback__
            )
        else:
            self.span_context.__exit__(None, None, None)


class _RecordingContext:
    """Manager of the creation of records from record calls.

    An instance of this class is produced when using an
    [App][trulens_eval.app.App] as a context mananger, i.e.:
    Example:
        ```python
        app = ...  # your app
        truapp: TruChain = TruChain(app, ...) # recorder for LangChain apps
        with truapp as recorder:
            app.invoke(...) # use your app
        recorder: RecordingContext
        ```

    Each instance of this class produces a record for every "root" instrumented
    method called. Root method here means the first instrumented method in a
    call stack. Note that there may be more than one of these contexts in play
    at the same time due to:
    - More than one wrapper of the same app.
    - More than one context manager ("with" statement) surrounding calls to the
      same app.
    - Calls to "with_record" on methods that themselves contain recording.
    - Calls to apps that use trulens internally to track records in any of the
      supported ways.
    - Combinations of the above.
    """

    def __init__(
        self,
        app: _WithInstrumentCallbacks,
        record_metadata: serial_utils.JSON = None,
        tracer: Optional[Tracer] = None,
        span: Optional[PhantomSpanRecordingContext] = None,
        span_ctx: Optional[SpanContext] = None,
    ):
        self.calls: Dict[
            types_schema.CallID, mod_record_schema.RecordAppCall
        ] = {}
        """A record (in terms of its RecordAppCall) in process of being created.

        Storing as a map as we want to override calls with the same id which may
        happen due to methods producing awaitables or generators. These result
        in calls before the awaitables are awaited and then get updated after
        the result is ready.
        """
        # TODEP: To deprecated after migration to span-based tracing.

        self.records: List[mod_record_schema.Record] = []
        """Completed records."""

        self.lock: Lock = Lock()
        """Lock blocking access to `records` when adding calls or
        finishing a record."""

        self.token: Optional[contextvars.Token] = None
        """Token for context management."""

        self.app: _WithInstrumentCallbacks = app
        """App for which we are recording."""

        self.record_metadata = record_metadata
        """Metadata to attach to all records produced in this context."""

        self.tracer: Optional[Tracer] = tracer
        """EXPERIMENTAL: otel-tracing

        OTEL-like tracer for recording.
        """

        self.span: Optional[PhantomSpanRecordingContext] = span
        """EXPERIMENTAL: otel-tracing

        Span that represents a recording context (the with block)."""

        self.span_ctx = span_ctx
        """EXPERIMENTAL: otel-tracing

        The context manager for the above span.
        """

    @property
    def spans(self) -> Dict[SpanContext, Span]:
        """Get the spans of the tracer in this context."""
        # EXPERIMENTAL: otel-tracing

        if self.tracer is None:
            return {}

        return self.tracer.spans

    def __iter__(self):
        return iter(self.records)

    def get(self) -> mod_record_schema.Record:
        """Get the single record only if there was exactly one or throw
        an error otherwise."""

        if len(self.records) == 0:
            raise RuntimeError("Recording context did not record any records.")

        if len(self.records) > 1:
            raise RuntimeError(
                "Recording context recorded more than 1 record. "
                "You can get them with ctx.records, ctx[i], or `for r in ctx: ...`."
            )

        return self.records[0]

    def __getitem__(self, idx: int) -> mod_record_schema.Record:
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def __hash__(self) -> int:
        # The same app can have multiple recording contexts.
        return hash(id(self.app)) + hash(id(self.records))

    def __eq__(self, other):
        return hash(self) == hash(other)
        # return id(self.app) == id(other.app) and id(self.records) == id(other.records)

    def add_call(self, call: mod_record_schema.RecordAppCall):
        """Add the given call to the currently tracked call list."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            # NOTE: This might override existing call record which happens when
            # processing calls with awaitable or generator results.
            self.calls[call.call_id] = call

    def finish_record(
        self,
        calls_to_record: Callable[
            [
                List[mod_record_schema.RecordAppCall],
                types_schema.Metadata,
                Optional[mod_record_schema.Record],
            ],
            mod_record_schema.Record,
        ],
        existing_record: Optional[mod_record_schema.Record] = None,
    ):
        """Run the given function to build a record from the tracked calls and any
        pre-specified metadata."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            record = calls_to_record(
                list(self.calls.values()), self.record_metadata, existing_record
            )
            self.calls = {}

            if existing_record is None:
                # If existing record was given, we assume it was already
                # inserted into this list.
                self.records.append(record)

        return record


class _WithInstrumentCallbacks:
    """Abstract definition of callbacks invoked by Instrument during
    instrumentation or when instrumented methods are called.

    Needs to be mixed into [App][trulens_eval.app.App].
    """

    # Called during instrumentation.
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """Callback to be called by instrumentation system for every function
        requested to be instrumented.

        Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).

            path: The path of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def get_method_path(self, obj: object, func: Callable) -> serial_utils.Lens:
        """Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.
        Args:
            func: The function to match.
        """

        raise NotImplementedError

    # Called after recording of an invocation.
    def on_new_root_span(
        self,
        ctx: _RecordingContext,
        root_span: LiveSpanCall,
    ) -> mod_record_schema.Record:
        """Called by instrumented methods if they are root calls (first instrumned
        methods in a call stack).

        Args:
            ctx: The context of the recording.

            root_span: The root span that was recorded.
        """
        # EXPERIMENTAL: otel-tracing

        raise NotImplementedError


class AppTracingCallbacks(TracingCallbacks[T]):
    """Extension to TracingCallbacks that keep track of apps that are
    instrumenting their constituent calls."""

    @classmethod
    def on_callable_wrapped(
        cls,
        wrapper: Callable[..., Any],
        app: _WithInstrumentCallbacks,
        **kwargs: Dict[str, Any],
    ):
        if not python_utils.safe_hasattr(wrapper, APPS):
            apps = set()
            setattr(wrapper, APPS, apps)
        else:
            apps = python_utils.safe_getattr(wrapper, APPS)

        apps.add(app)

        return super().on_callable_wrapped(wrapper, **kwargs)

    def __init__(
        self,
        app: _WithInstrumentCallbacks,
        span_type: Type[Span] = LiveSpanCall,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(span_type=span_type, **kwargs)

        self.app = app
        self.apps = python_utils.safe_getattr(self.wrapper, APPS)
