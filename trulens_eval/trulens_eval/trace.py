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
from typing import Any, Dict, List, Optional
import uuid

import pydantic

from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import record as mod_record_schema

logger = logging.getLogger(__name__)

class Context(pydantic.BaseModel):
    """Identifiers for a span."""

    trace_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    """Unique identifier for the trace.
    
    Each root span has a unique trace id."""

    span_id: int = pydantic.Field(default_factory=lambda: random.getrandbits(64))
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
        return self.trace_id == other.trace_id and self.span_id == other.span_id

class Span(pydantic.BaseModel):
    """A span of observed time in the application."""

    context: Context
    """Identifiers."""

    parent: Optional[Context] = None
    """Optional parent identifier."""

    error: Optional[str] = None
    """Optional error message if the observed computation raised an exception."""

class SpanRecord(Span):
    """Track an entire Record creation."""

    record: Optional[mod_record_schema.Record] = None

class SpanMethodCall(Span):
    """Track a method call."""

    call: Optional[mod_record_schema.RecordAppCall] = None

class SpanCost(Span):
    """Track costs of some computation."""

    cost: Optional[mod_base_schema.Cost] = None
    endpoint: Optional[Any] = None # TODO: Type

    def tally(self):
        for span in self.tracer.spans.values():
            if span.context == self.context:
                continue

            if span.cost is not None:
                self.cost += span.cost
        

class Tracer():
    def __init__(self, context: Optional[Context] = None):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=context)
        self.trace_id = uuid.uuid4()
        self._spans: Dict[Context, Span] = {}

    @contextlib.contextmanager
    def _span(self, cls):
        print("tracer", cls.__name__)
        context = Context(trace_id=self.trace_id)
        span = cls(context=context, tracer=self, parent=self.context.get())
        self._spans[context] = span

        token = self.context.set(context)

        try:
            yield span
        except BaseException as e:
            span.error = str(e) + "\n\n" + traceback.format_exc()
        finally:
            self.context.reset(token)
            return

    def record(self):
        return self._span(SpanRecord)
    
    def method(self):
        return self._span(SpanMethodCall)
    
    def cost(self):
        return self._span(SpanCost)

    @property
    def spans(self):
        return self._spans

class NullTracer(Tracer):
    """Tracer that does not save the spans it makes."""

    def __init__(self, context: Optional[Context] = None):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=context)
        self.trace_id = uuid.uuid4()

    @contextlib.contextmanager
    def _span(self, cls):
        print("null", cls.__name__)
        context = Context(trace_id=self.trace_id)
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
        with self.tracer.record() as root:
            tok = self.context.set(root.context)
            yield self.tracer

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
