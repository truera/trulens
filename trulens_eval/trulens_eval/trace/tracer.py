"""Tracer, manages spans."""

from __future__ import annotations

import contextvars
from logging import getLogger
import random
from typing import Iterator, Mapping, Optional, Type

import opentelemetry
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
from opentelemetry.util._decorator import _agnosticcontextmanager
import pydantic

from trulens_eval import trace as mod_trace
from trulens_eval.trace import span as mod_span

logger = getLogger(__name__)

def trace_id_of_string_id(s: str) -> mod_trace.TTraceID:
    """Convert a string id to a trace ID.
    
    Not an OT requirement.
    """

    return hash(s) % (1 << mod_trace.NUM_TRACEID_BITS)

def span_id_of_string_id(s: str) -> mod_trace.TSpanID:
    """Convert a string id to a span ID.
    
    Not an OT requirement.
    """

    return hash(s) % (1 << mod_trace.NUM_SPANID_BITS)

class Tracer(pydantic.BaseModel, ot_trace.Tracer):
    """Implementation of OpenTelemetry Tracer requirements."""

    model_config = {
        'arbitrary_types_allowed': True,
        'use_attribute_docstrings': True
    }
    """Pydantic configuration."""

    stack: contextvars.ContextVar[mod_trace.HashableSpanContext] = pydantic.Field(
        default_factory=lambda: contextvars.ContextVar("stack", default=None),
        exclude=True
    )

    instrumenting_module_name: str = "trulens_eval"
    instrumenting_library_version: Optional[str] = None#trulens_eval.__version__

    spans: mod_trace.ContextMapping[
        Mapping[str, ot_types.AttributeValue],
    ] = pydantic.Field(default_factory=dict)
    """Spans recorded by the tracer."""

    state: ot_span.TraceState = pydantic.Field(default_factory=ot_span.TraceState)
    """Trace attributes."""

    trace_id: mod_trace.TTraceID
    """Unique identifier for the trace."""

    def __init__(
        self,
        trace_id: Optional[mod_trace.TTraceId] = None,
        **kwargs
    ):
        if trace_id is None:
            trace_id = random.getrandbits(mod_trace.NUM_TRACEID_BITS)

        kwargs['trace_id'] = trace_id

        super().__init__(**kwargs)

    def new_span(
        self,
        name: str,
        cls: Type[mod_span.Span],
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_trace.types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True
    ) -> mod_trace.Span:
        """See [new_span][opentelemetry.trace.Tracer.new_span]."""

        span_context = mod_trace.HashableSpanContext(
            trace_id=self.trace_id,
            span_id=random.getrandbits(mod_trace.NUM_SPANID_BITS),
            is_remote=False,
            trace_state = self.state
        )

        span = cls(
            name=name,
            context=span_context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception
        )

        if context is not None:
            context = mod_trace.make_hashable(context)
            span.add_link(context, {mod_span.Span.vendor_attr("relationship"): "parent"})

        self.spans[span_context] = span

        return span

    def start_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_trace.types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> mod_span.Span:
        """See [start_span][opentelemetry.trace.Tracer.start_span]."""

        if context is None:
            parent_context = self.stack.get()

        else:
            parent_context = mod_trace.make_hashable(context)

            if parent_context.trace_id != self.trace_id:
                logger.warning("Parent context is not being traced by this tracer.")

        span_context = mod_trace.HashableSpanContext(
            trace_id=self.trace_id,
            span_id=random.getrandbits(mod_trace.NUM_SPANID_BITS),
            is_remote=False,
            trace_state = self.state # unsure whether these should be shared across all spans produced by this tracer
        )

        span = mod_span.SpanUntyped(
            name=name,
            context=span_context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception
        )

        if parent_context is not None:
            span.add_link(parent_context, {mod_span.Span.vendor_attr("relationship"): "parent"})

        self.spans[span_context] = span

        return span

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = opentelemetry.trace.SpanKind.INTERNAL,
        attributes: ot_types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[mod_span.Span]:
        """See [start_as_current_span][opentelemetry.trace.Tracer.start_as_current_span]."""

        if context is not None:
            context = mod_trace.make_hashable(context)

        span = self.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception
        )

        token = self.stack.set(span.context)

        # Unsure if this ot_trace stuff is needed.
        span_token = ot_trace.use_span(span, end_on_exit=end_on_exit).__enter__()
        yield span

        # Same
        span_token.__exit__(None, None, None)

        self.stack.reset(token)
        return
