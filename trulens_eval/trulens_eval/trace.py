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
from typing import Dict, Optional
import uuid

import pydantic

from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import record as mod_record_schema

logger = logging.getLogger(__name__)

class Context(pydantic.BaseModel):
    trace_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    span_id: int = pydantic.Field(default_factory=lambda: random.getrandbits(64))

    def __str__(self):
        return f"{self.trace_id.int % 0xff:02x}/{self.span_id % 0xff:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.trace_id) + hash(self.span_id)
    
    def __eq__(self, other):
        return self.trace_id == other.trace_id and self.span_id == other.span_id

class Span(pydantic.BaseModel):
    context: Context
    parent: Optional[Context] = None
    error: Optional[str] = None

class SpanRecord(Span):
    record: Optional[mod_record_schema.Record] = None

class SpanMethodCall(Span):
    call: Optional[mod_record_schema.RecordAppCall] = None

class SpanCost(Span):
    cost: Optional[mod_base_schema.Cost] = None

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
            span.error = str(e)
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
