"""Implementation of recording that resembles the tracing process in
OpenTelemetry.

This module is likely temporary and will be replaced by actual OpenTelemetry sdk
components or implementations that are compatible with its API.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import random
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Type
import uuid

import pydantic

# import trulens_eval.app as mod_app
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.utils import pyschema as mod_pyschema
from trulens_eval.utils.serial import Lens

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

    context: Context
    """Identifiers."""

    parent: Optional[Context] = None
    """Optional parent identifier."""

    error: Optional[str] = None
    """Optional error message if the observed computation raised an exception."""

    @property
    def tracer(self):
        return self.context.tracer

    def iter_children(self, transitive: bool = True) -> Iterable[Span]:
        """Iterate over all spans that are children of this span."""

        # TODO: runtime
        for span in self.tracer.spans.values():
            if span.parent == self.context:
                yield span
                if transitive:
                    yield from span.iter_children(transitive=transitive)

    def iter_family(self) -> Iterable[Span]:
        """Iterate itself and all children transitively."""

        yield self
        yield from self.iter_children()

    def is_root(self) -> bool:
        """Check if this span is a "root" span here meaning that it is either a
        SpanRecord or its parent is a SpanRecord."""

        if self.parent is None:
            return True

        parent = self.context.tracer.spans.get(self.parent)
        return parent.is_root()

    def cost_tally(self) -> mod_base_schema.Cost:
        total = mod_base_schema.Cost()

        for span in self.iter_family():
            if isinstance(span, SpanCost) and span.cost is not None:
                total += span.cost

        return total


class SpanAppRecordingContext(Span):
    """Tracks the context of an app used as a context manager."""

    # record: Optional[mod_record_schema.Record] = None
    ctx: Optional[Any] = None  # TODO: Type

    def is_root(self) -> bool:
        return True


class LiveSpanCall(Span):
    """Track a function call.
    
    WARNING:
        This span contains references to live objects. These may change after
        this span is created but before it is dumped or exported.
    """

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

    func_name: Optional[str] = None
    """Function name."""

    args: Optional[Dict[str, Any]] = None
    """Arguments to the function call.
    
    WARNING: Contains references to live objects.
    """

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


class SpanCost(Span):
    """Track costs of some computation."""

    cost: Optional[mod_base_schema.Cost] = None
    endpoint: Optional[Any] = None  # TODO: Type


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
        obj_to_path_map: Dict[int, Lens] = {},
        stack: List[mod_record_schema.RecordAppCallMethod] = []
    ):
        # TODO: what if it is not a method call?

        frame_ident = mod_record_schema.RecordAppCallMethod(
            path=obj_to_path_map.get(id(span.obj), None),
            method=mod_pyschema.Method.of_method(
                span.func, obj=span.obj, cls=span.cls
            )
        )

        stack = stack + [frame_ident]
        span.stack = stack

        for subspan in span.iter_children(transitive=False):
            if not isinstance(subspan, LiveSpanCall):
                continue

            Tracer._fill_stacks(subspan, stack=stack)

    def _call_of_spancall(
        self, span: LiveSpanCall
    ) -> mod_record_schema.RecordAppCall:
        """Convert a SpanCall to a RecordAppCall."""

        return mod_record_schema.RecordAppCall(
            call_id=span.call_id,
            stack=span.stack,
            args=span.args,
            rets=span.ret,
            error=span.error,
            perf=span.perf,
            pid=span.pid,
            tid=span.tid
        )

    def records_of_recording(
        self, recording: SpanAppRecordingContext
    ) -> Iterable[mod_record_schema.Record]:
        """Convert a recording based on spans to a list of records."""

        obj_id_to_path_map = {}

        for root_span in recording.iter_children(transitive=False):
            assert isinstance(root_span, LiveSpanCall)
            self._fill_stacks(root_span, obj_to_path_map=obj_id_to_path_map)

            root_perf = root_span.perf
            root_cost = root_span.cost_tally()

            calls = [self._call_of_spancall(root_span)]
            for span in root_span.iter_children():
                if not isinstance(span, LiveSpanCall):
                    continue
                calls.append(self._call_of_spancall(span))

            record = mod_record_schema.Record(
                record_id=str(uuid.uuid4()),
                app_id=str(uuid.uuid4()),
                main_input="placeholder",
                main_output="placeholder",
                main_error=None,
                calls=calls,
                perf=root_perf,
                cost=root_cost
            )
            yield record

    @contextlib.contextmanager
    def _span(self, cls):
        print("tracer", cls.__name__)
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(context=context, tracer=self, parent=self.context.get())
        self.spans_[context] = span

        token = self.context.set(context)

        try:
            yield span
        except BaseException as e:
            span.error = str(e) + "\n\n" + traceback.format_exc()
        finally:
            self.context.reset(token)
            return

    def recording(self):
        return self._span(SpanAppRecordingContext)

    def method(self):
        return self._span(LiveSpanCall)

    def cost(self):
        return self._span(SpanCost)

    @property
    def spans(self):
        return self.spans_


class NullTracer(Tracer):
    """Tracer that does not save the spans it makes."""

    @contextlib.contextmanager
    def _span(self, cls):
        print("null", cls.__name__)
        context = Context(trace_id=self.trace_id, tracer=self)
        span = cls(context=context, tracer=self, parent=self.context.get())
        token = self.context.set(context)

        try:
            yield span
        except BaseException as e:
            # ignore exception since spans are also ignored/not recorded
            pass
        finally:
            self.context.reset(token)
            return

    @property
    def spans(self):
        return []


class TracerProvider():

    def __init__(self):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=None)

        self.tracer: Tracer = NullTracer()

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
