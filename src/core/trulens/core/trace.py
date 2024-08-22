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
import inspect
import logging
import os
import random
import threading as th
from threading import Lock
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import uuid

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import opentelemetry.trace as ot_trace
from opentelemetry.trace import status as trace_status
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.utils import wrap as wrap_utils

if TYPE_CHECKING:
    from trulens.core import otel as mod_otel
    from trulens.core.schema import record as mod_record_schema
    from trulens.core.schema import types as types_schema
    from trulens.core.utils import json as json_utils
    from trulens.core.utils import pyschema as pyschema_utils
    from trulens.core.utils import python as python_utils
    from trulens.core.utils import serial as serial_utils


logger = logging.getLogger(__name__)

INSTRUMENT: str = "__tru_instrumented"
"""Attribute name to be used to flag instrumented objects/methods/others."""

APPS: str = "__tru_apps"
"""Attribute name for storing apps that expect to be notified of calls."""


class Context(pydantic.BaseModel):
    """Identifiers for a span."""

    trace_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    """Unique identifier for the trace.

    Each root span has a unique trace id."""

    span_id: int = pydantic.Field(
        default_factory=lambda: random.getrandbits(64)
    )
    """Identifier for the span.

    Meant to be at least unique within the same trace_id.
    """

    tracer: Tracer = pydantic.Field(exclude=True)
    """Reference to the tracer that created this span."""

    def __str__(self):
        return f"{self.trace_id.int % 0xFF:02x}/{self.span_id % 0xFF:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.trace_id) + hash(self.span_id)

    def __eq__(self, other):
        if other is None:
            return False
        return self.trace_id == other.trace_id and self.span_id == other.span_id


class Span(pydantic.BaseModel):
    """A span of observed time in the application."""

    model_config = {"arbitrary_types_allowed": True}

    context: Context = pydantic.Field(exclude=True)
    """Identifiers."""

    parent: Optional[Context] = pydantic.Field(None, exclude=True)
    """Optional parent identifier."""

    error: Optional[Exception] = pydantic.Field(None, exclude=True)
    """Optional error if the observed computation raised an exception."""

    start_timestamp: Optional[int] = pydantic.Field(None, exclude=True)
    """Start time in nanoseconds since epoch."""

    end_timestamp: Optional[int] = pydantic.Field(None, exclude=True)
    """End time in nanoseconds since epoch.

    None if not yet finished."""

    def __str__(self):
        return f"{type(self).__name__} {self.context} -> {self.parent}"

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

        # TODO: runtime
        for span in self.tracer.spans.values():
            if span.parent == self.context:
                if isinstance(span, PhantomSpan) and not include_phantom:
                    # Note that transitive being false is ignored if phantom is skipped.
                    yield from span.iter_children(
                        transitive=transitive, include_phantom=include_phantom
                    )
                else:
                    yield span
                    if transitive:
                        yield from span.iter_children(
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
    """Methods for converting a span to an OTEL span.

    !!! Warning
        This is an experimental feature. OTEL integration is ongoing.
    """

    @staticmethod
    def otel_context_of_context(context: Context) -> ot_span.SpanContext:
        return ot_span.SpanContext(
            trace_id=hash(context.trace_id.int) & ot_span._TRACE_ID_MAX_VALUE,
            span_id=hash(context.span_id) & ot_span._SPAN_ID_MAX_VALUE,
            is_remote=False,
        )

    def otel_name(self) -> str:
        return "unnamed"

    def otel_context(self) -> ot_types.SpanContext:
        return self.otel_context_of_context(self.context)

    def otel_parent_context(self) -> Optional[ot_types.SpanContext]:
        if self.parent is None:
            return None
        return self.otel_context_of_context(self.parent)

    def attributes(self):
        return self.model_dump()

    def otel_attributes(self) -> ot_types.Attributes:
        return mod_otel.flatten_lensed_attributes(self.attributes())

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
            self.recording.app.app_id if self.recording is not None else None
        )

        return ret

    def finish(self):
        super().finish()

        assert self.recording is not None

        app = self.recording.app

        for span in self.iter_children(transitive=False):
            if not isinstance(span, LiveSpanCall):
                continue
            app.on_new_root_span(recording=self.recording, root_span=span)

        app.on_new_recording_span(recording_span=self)

    async def afinish(self):
        await super().afinish()

        assert self.recording is not None

        app = self.recording.app

        for span in self.iter_children(transitive=False):
            if not isinstance(span, LiveSpanCall):
                continue
            app.on_new_root_span(recording=self.recording, root_span=span)

        app.on_new_recording_span(recording_span=self)

    def otel_name(self) -> str:
        return f"PhantomSpanRecordingContext({self.recording.app.app_id if self.recording is not None else None})"


class SpanCall(OTELExportable):
    """Non-live fields of a function call span."""

    model_config = {"arbitrary_types_allowed": True}

    call_id: Optional[uuid.UUID] = pydantic.Field(None, exclude=True)
    """Unique identifier for the call."""

    stack: Optional[List[mod_record_schema.RecordAppCallMethod]] = (
        pydantic.Field(None, exclude=True)
    )
    """Call stack of instrumented methods only."""

    sig: Optional[inspect.Signature] = pydantic.Field(None, exclude=True)
    """Signature of the function."""

    func_name: Optional[str] = None
    """Function name."""

    pid: Optional[int] = pydantic.Field(None, exclude=True)
    """Process id."""

    tid: Optional[int] = None
    """Thread id."""

    def attributes(self):
        ret = super().attributes()

        ret["sig"] = str(self.sig)
        ret["call_id"] = str(self.call_id)
        ret["stack"] = json_utils.jsonify(self.stack)

        return ret

    def otel_attributes(self) -> ot_types.Attributes:
        # temp = {f"trulens_eval@{k}": v for k, v in self.attributes().items()}
        return mod_otel.flatten_lensed_attributes(self.attributes())

    def otel_resource_attributes(self) -> Dict[str, Any]:
        ret = super().otel_resource_attributes()

        ret[ResourceAttributes.PROCESS_PID] = self.pid
        ret["thread.id"] = self.tid  # TODO: semconv

        return ret

    def otel_name(self) -> str:
        return f"{self.__class__.__name__}({self.func_name})"


class LiveSpanCall(LiveSpan, SpanCall):
    """Track a function call."""

    model_config = {"arbitrary_types_allowed": True}

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

    Excluisve with `error`.
    """

    live_error: Optional[Any] = pydantic.Field(None, exclude=True)
    """Error raised by the function call.

    Exclusive with `ret`.
    """


class WithCost(LiveSpan):
    """Mixin to indicate the span has costs tracked."""

    cost: base_schema.Cost = pydantic.Field(
        default_factory=base_schema.Cost, exclude=True
    )
    """Cost of the computation spanned."""

    endpoint: Optional[Any] = pydantic.Field(None, exclude=True)
    """Endpoint handling cost extraction for this span/call."""

    def attributes(self):
        ret = super().attributes()

        ret["cost"] = self.cost.model_dump()

        return ret

    def __init__(self, cost: Optional[base_schema.Cost] = None, **kwargs):
        if cost is None:
            cost = base_schema.Cost()

        super().__init__(cost=cost, **kwargs)


class LiveSpanCallWithCost(LiveSpanCall, WithCost):
    pass


class Tracer(pydantic.BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    context: contextvars.ContextVar[Optional[Context]] = pydantic.Field(
        default=None, exclude=True
    )

    trace_id: uuid.UUID

    spans_: Dict[Context, Span] = pydantic.Field(
        default_factory=dict, exclude=True
    )

    def __init__(self, context: Optional[Context] = None):
        super().__init__(
            context=contextvars.ContextVar("context", default=context),
            trace_id=uuid.uuid4(),
            spans_={},
        )

    @staticmethod
    def _fill_stacks(
        span: LiveSpanCall,
        get_method_path: Callable,
        stack: List[mod_record_schema.RecordAppCallMethod] = [],
    ):
        # TODO: what if it is not a method call?

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
            if not isinstance(subspan, LiveSpanCall):
                continue

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
            rets=span.live_ret,
            error=str(span.live_error),
            perf=base_schema.Perf.of_ns_timestamps(
                start_ns_timestamp=span.start_timestamp,
                end_ns_timestamp=span.end_timestamp,
            ),
            pid=span.pid,
            tid=span.tid,
        )

    def record_of_root_span(self, recording: Any, root_span: LiveSpanCall):
        """Convert a root span to a record."""

        assert isinstance(root_span, LiveSpanCall)

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

        root_cost = root_span.total_cost()

        calls = [self._call_of_spancall(root_span)]
        for span in root_span.iter_children():
            if not isinstance(span, LiveSpanCall):
                continue
            calls.append(self._call_of_spancall(span))

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
        )

        # record_id determinism
        record.record_id = json_utils.obj_id_of_obj(record, prefix="record")

        return record

    def records_of_recording(
        self, recording: PhantomSpanRecordingContext
    ) -> Iterable[mod_record_schema.Record]:
        """Convert a recording based on spans to a list of records."""

        for root_span in recording.iter_children(transitive=False):
            yield self.record_of_root_span(recording, root_span)

    @contextlib.contextmanager
    def _span(self, cls, **kwargs):
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(
            context=context, tracer=self, parent=self.context.get(), **kwargs
        )
        self.spans_[context] = span

        token = self.context.set(context)

        try:
            yield span
        except BaseException as e:
            span.error = e
        finally:
            self.context.reset(token)
            span.finish()
            if span.error is not None:
                raise span.error

    @contextlib.asynccontextmanager
    async def _aspan(self, cls, **kwargs):
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(
            context=context, tracer=self, parent=self.context.get(), **kwargs
        )
        self.spans_[context] = span

        token = self.context.set(context)

        try:
            yield span
        except BaseException as e:
            span.error = e
        finally:
            self.context.reset(token)
            await span.afinish()
            if span.error is not None:
                raise span.error

    def recording(self):
        return self._span(PhantomSpanRecordingContext)

    def method(self):
        return self._span(LiveSpanCall)

    def cost(self, cost: Optional[base_schema.Cost] = None):
        return self._span(LiveSpanCallWithCost, cost=cost)

    def phantom(self):
        return self._span(PhantomSpan)

    async def arecording(self):
        return self._aspan(PhantomSpanRecordingContext)

    async def amethod(self):
        return self._aspan(LiveSpanCall)

    async def acost(self, cost: Optional[base_schema.Cost] = None):
        return self._aspan(LiveSpanCallWithCost, cost=cost)

    async def aphantom(self):
        return self._aspan(PhantomSpan)

    @property
    def spans(self):
        return self.spans_


class NullTracer(Tracer):
    """Tracer that does not save the spans it makes."""

    @contextlib.contextmanager
    def _span(self, cls):
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(context=context, tracer=self, parent=self.context.get())
        token = self.context.set(context)

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
    async def _aspan(self, cls):
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(context=context, tracer=self, parent=self.context.get())
        token = self.context.set(context)

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


class TracerProvider:
    def __init__(self):
        self.context: contextvars.ContextVar[Optional[Context]] = (
            contextvars.ContextVar("context", default=None)
        )

        self.tracer: Tracer = Tracer()

    @contextlib.contextmanager
    def trace(self):
        prior_tracer = self.tracer

        self.tracer = Tracer(context=self.context.get())
        with self.tracer.recording() as root:
            tok = self.context.set(root.context)
            yield root

        self.context.reset(tok)

        self.tracer = prior_tracer

    def get_tracer(self):
        return self.tracer


tracer_provider = TracerProvider()
"""Global tracer provider.
All traces are mady by this provider.
"""


def get_tracer():
    return tracer_provider.get_tracer()


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

        tracer = get_tracer()

        if not issubclass(span_type, LiveSpanCall):
            raise ValueError("span_type must be a subclass of LiveSpanCall.")

        self.span_context = tracer._span(span_type)
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


class RecordingContext:
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
        app: WithInstrumentCallbacks,
        record_metadata: serial_utils.JSON = None,
        tracer: Optional[Tracer] = None,
        span: Optional[PhantomSpanRecordingContext] = None,
        span_ctx: Optional[Context] = None,
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

        self.app: WithInstrumentCallbacks = app
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
    def spans(self) -> Dict[Context, Span]:
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


class WithInstrumentCallbacks:
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
        ctx: RecordingContext,
        root_span: Span,
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
        app: WithInstrumentCallbacks,
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
        app: WithInstrumentCallbacks,
        span_type: Type[Span] = LiveSpanCall,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(span_type=span_type, **kwargs)

        self.app = app
        self.apps = python_utils.safe_getattr(self.wrapper, APPS)
