"""
Modules in this folder depend on each other which makes it impossible to import
them without circular import errors. Because of this, some imports need to be
put into `if TYPE_CHECKING` blocks and classes that depend on those imports need
to be "rebuilt" with `model_rebuild`. This only applies to `pydantic.BaseModel`
classes. Type hints on non-pydantic classes are never interpreted hence no need
to "rebuild" those.
"""

from . import context as core_context
from . import otel as core_otel
from . import sem as core_sem
from . import span as core_span
from . import trace as core_trace

core_context.SpanContext.model_rebuild()
core_context.TraceState.model_rebuild()
core_otel.Span.model_rebuild()
core_otel.Tracer.model_rebuild()
core_otel.TracerProvider.model_rebuild()
core_span.Span.model_rebuild()
core_sem.TypedSpan.model_rebuild()
core_trace.Tracer.model_rebuild()
core_trace.TracerProvider.model_rebuild()
