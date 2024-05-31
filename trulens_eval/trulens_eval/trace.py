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

from trulens_eval.schema import record as mod_record_schema

logger = logging.getLogger(__name__)

class Context(pydantic.BaseModel):
    trace_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    span_id: int = pydantic.Field(default_factory=lambda: random.getrandbits(64))

    def __str__(self):
        return f"{self.trace_id.int % 0xff:02x}/{self.span_id % 0xff:02x})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.trace_id) + hash(self.span_id)
    
    def __eq__(self, other):
        return self.trace_id == other.trace_id and self.span_id == other.span_id

class Span(pydantic.BaseModel):
    context: Context
    parent: Optional[Context] = None

class SpanRecord(Span):
    record: Optional[mod_record_schema.Record] = None

class SpanMethodCall(Span):
    call: Optional[mod_record_schema.RecordAppCall] = None

class Tracer():
    def __init__(self, context: Optional[Context] = None):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=context)
        self.trace_id = uuid.uuid4()
        self.spans: Dict[Context, Span] = {}

    @contextlib.contextmanager
    def record(self):
        context = Context(trace_id=self.trace_id)
        span = SpanRecord(context=context, tracer=self, parent=self.context.get())
        self.spans[context] = span

        token = self.context.set(context)

        yield span

        self.context.reset(token)

        return

    @contextlib.contextmanager
    def method(self):
        context = Context(trace_id=self.trace_id)
        span = SpanMethodCall(context=context, tracer=self, parent=self.context.get())
        self.spans[context] = span

        token = self.context.set(context)

        yield span

        self.context.reset(token)

        return

class TracerProvider():
    def __init__(self):
        self.context: contextvars.ContextVar[Optional[Context]] = \
            contextvars.ContextVar("context", default=None)

    @contextlib.contextmanager
    def trace(self):
        tracer = Tracer(context=self.context.get())
        with tracer.record() as root:
            tok = self.context.set(root.context)
            yield tracer

        self.context.reset(tok)