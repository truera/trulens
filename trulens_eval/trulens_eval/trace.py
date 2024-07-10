"""Implementation of recording that resembles the tracing process in
OpenTelemetry.

This module is likely temporary and will be replaced by actual OpenTelemetry sdk
components or implementations that are compatible with its API.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import random
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import uuid

import pydantic

# import trulens_eval.app as mod_app # circular import issues
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.utils import pyschema as mod_pyschema
from trulens_eval.utils import json as mod_json_utils

logger = logging.getLogger(__name__)


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
        return f"{self.trace_id.int % 0xff:02x}/{self.span_id % 0xff:02x}"

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

    model_config = {'arbitrary_types_allowed': True}

    context: Context
    """Identifiers."""

    parent: Optional[Context] = None
    """Optional parent identifier."""

    error: Optional[Exception] = None
    """Optional error if the observed computation raised an exception."""

    def __str__(self):
        return f"{type(self).__name__} {self.context} -> {self.parent}"

    @property
    def tracer(self):
        return self.context.tracer

    def finish(self):
        pass

    async def afinish(self):
        return self.finish()

    def iter_children(
        self,
        transitive: bool = True,
        include_phantom: bool = False
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
                            include_phantom=include_phantom
                        )

    def iter_family(self, include_phantom: bool = False) -> Iterable[Span]:
        """Iterate itself and all children transitively."""

        if (not isinstance(self, PhantomSpan)) or include_phantom:
            yield self

        yield from self.iter_children(
            include_phantom=include_phantom, transitive=True
        )

    def is_root(self) -> bool:
        """Check if this span is a "root" span here meaning that it is either a
        SpanRecord or its parent is a SpanRecord."""

        if self.parent is None:
            return True

        parent = self.context.tracer.spans.get(self.parent)
        return parent.is_root()

    def total_cost(self) -> mod_base_schema.Cost:
        """Total costs of this span and all its transitive children."""

        total = mod_base_schema.Cost()

        for span in self.iter_family(include_phantom=True):
            if isinstance(span, PhantomSpanCost) and span.cost is not None:
                total += span.cost

        return total


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


class PhantomSpanRecordingContext(PhantomSpan):
    """Tracks the context of an app used as a context manager."""

    # record: Optional[mod_record_schema.Record] = None
    recording: Optional[
        Any] = None  # TODO: app.RecordingContext # circular import issues

    def is_root(self) -> bool:
        return True

    def finish(self):
        app = self.recording.app

        for span in self.iter_children(transitive=False):
            if not isinstance(span, LiveSpanCall):
                continue
            app.on_new_root_span(recording=self.recording, root_span=span)


class LiveSpanCall(LiveSpan):
    """Track a function call.
    
    WARNING:
        This span contains references to live objects. These may change after
        this span is created but before it is dumped or exported.
    """

    model_config = {'arbitrary_types_allowed': True}

    call_id: Optional[uuid.UUID] = None

    stack: Optional[List[mod_record_schema.RecordAppCallMethod]] = None

    # call: Optional[mod_record_schema.RecordAppCall] = None
    obj: Optional[Any] = None
    """Self object if method call.
    
    WARNING: This is a live object.
    """

    cls: Optional[Type] = None
    """Class if method/static/class method call."""

    func: Optional[Callable] = None
    """Function object.
    
    WARNING: This is a live object.
    """

    sig: Optional[inspect.Signature] = None
    """Signature of the function."""

    func_name: Optional[str] = None
    """Function name."""

    args: Optional[Tuple[Any, ...]] = None
    """Positional arguments to the function call.
    
    WARNING: Contains references to live objects.
    """

    kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments to the function call.

    WARNING: Contains references to live objects.
    """

    bindings: Optional[inspect.BoundArguments] = None
    """Bound arguments to the function call if can be bound."""

    ret: Optional[Any] = None
    """Return value of the function call.
    
    Excluisve with `error`.

    WARNING: This is a live object.
    """

    error: Optional[Any] = None
    """Error raised by the function call.
    
    Exclusive with `ret`.
    """

    perf: Optional[mod_base_schema.Perf] = None
    """Performance metrics."""

    pid: Optional[int] = None
    """Process id."""

    tid: Optional[int] = None
    """Thread id."""


class PhantomSpanCost(PhantomSpan):
    """Track costs of some computation."""

    cost: Optional[mod_base_schema.Cost
                  ] = pydantic.Field(default_factory=mod_base_schema.Cost)
    endpoint: Optional[Any] = None  # TODO: Type

    def __init__(self, cost: Optional[mod_base_schema.Cost] = None, **kwargs):
        if cost is None:
            cost = mod_base_schema.Cost()

        super().__init__(cost=cost, **kwargs)


class Tracer(pydantic.BaseModel):
    model_config = {'arbitrary_types_allowed': True}

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
            spans_={}
        )

    @staticmethod
    def _fill_stacks(
        span: LiveSpanCall,
        get_method_path: Callable,
        stack: List[mod_record_schema.RecordAppCallMethod] = []
    ):
        # TODO: what if it is not a method call?

        path = get_method_path(obj=span.obj, func=span.func)

        frame_ident = mod_record_schema.RecordAppCallMethod(
            path=path,
            method=mod_pyschema.Method.of_method(
                span.func, obj=span.obj, cls=span.cls
            )
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

        args = dict(
            span.bindings.arguments
        ) if span.bindings is not None else None
        if args is not None:
            if "self" in args:
                del args["self"]  # remove self

        return mod_record_schema.RecordAppCall(
            call_id=str(span.call_id),
            stack=span.stack,
            args=args,
            rets=span.ret,
            error=str(span.error),
            perf=span.perf,
            pid=span.pid,
            tid=span.tid
        )

    def record_of_root_span(self, recording: Any, root_span: LiveSpanCall):
        """Convert a root span to a record."""

        assert isinstance(root_span, LiveSpanCall)

        app = recording.app

        self._fill_stacks(root_span, get_method_path=app.get_method_path)

        root_perf = root_span.perf
        root_cost = root_span.total_cost()

        calls = [self._call_of_spancall(root_span)]
        for span in root_span.iter_children():
            if not isinstance(span, LiveSpanCall):
                continue
            calls.append(self._call_of_spancall(span))

        bindings = root_span.bindings
        main_error = root_span.error

        if bindings is not None:
            main_input = app.main_input(
                func=root_span.func,
                sig=root_span.sig,
                bindings=root_span.bindings
            )
            if main_error is None:
                main_output = app.main_output(
                    func=root_span.func,
                    sig=root_span.sig,
                    bindings=root_span.bindings,
                    ret=root_span.ret
                )
            else:
                main_output = None
        else:
            main_input = None
            main_output = None

        record = mod_record_schema.Record(
            record_id="placeholder",
            app_id=app.app_id,
            main_input=mod_json_utils.jsonify(main_input),
            main_output=mod_json_utils.jsonify(main_output),
            main_error=mod_json_utils.jsonify(main_error),
            calls=calls,
            perf=root_perf,
            cost=root_cost
        )

        # record_id determinism
        record.record_id = mod_json_utils.obj_id_of_obj(record, prefix="record")

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
        print("tracer", cls.__name__)
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

    def cost(self, cost: Optional[mod_base_schema.Cost] = None):
        return self._span(PhantomSpanCost, cost=cost)

    def phantom(self):
        return self._span(PhantomSpan)
    
    async def arecording(self):
        return self._aspan(PhantomSpanRecordingContext)

    async def amethod(self):
        return self._aspan(LiveSpanCall)

    async def acost(self, cost: Optional[mod_base_schema.Cost] = None):
        return self._aspan(PhantomSpanCost, cost=cost)

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


class TracerProvider():

    def __init__(self):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=None)

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
