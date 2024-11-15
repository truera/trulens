"""
Modules in this folder depend on each other than makes it impossible to import
them linearly. Because of this, they need to be imported here and the delayed
typing information needs to be filled in afterwards with model_rebuild`.
"""

from . import context as core_context
from . import otel as core_otel
from . import sem as core_sem
from . import span as core_span
from . import trace as core_trace

core_trace.Tracer.model_rebuild()
core_span.Span.model_rebuild()
core_otel.Span.model_rebuild()
core_context.SpanContext.model_rebuild()
core_context.TraceState.model_rebuild()
core_sem.TypedSpan.model_rebuild()
core_trace.TracerProvider.model_rebuild()
core_otel.Tracer.model_rebuild()
core_otel.TracerProvider.model_rebuild()
