"""Span types.

These are roughly equivalent to RecordAppCall but abstract away specific method
information into type of call related to types of components.
"""

from typing import Dict, Mapping, Optional, Sequence
import pydantic
import opentelemetry.trace.span as ot_span

class Span(pydantic.BaseModel, ot_span.Span):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    # begin ot_span.Span requirements

    def end(self, end_time: Optional[int]=None):
        """See [end][opentelemetry.trace.span.Span.end]"""
        
        return super().end(end_time)

    def get_span_context(self) -> ot_span.SpanContext:
        return super().get_span_context()

    def set_attributes(self, attributes: Dict[str, str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]]) -> None:
        return super().set_attributes(attributes)

    def set_attribute(self, key: str, value: str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]) -> None:
        return super().set_attribute(key, value)

    def add_event(self, name: str, attributes: Mapping[str, str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]] | None = None, timestamp: int | None = None) -> None:
        return super().add_event(name, attributes, timestamp)

    def add_link(self, context: ot_span.SpanContext, attributes: Mapping[str, str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]] | None = None) -> None:
        return super().add_link(context, attributes)

    def update_name(self, name: str) -> None:
        return super().update_name(name)

    def is_recording(self) -> bool:
        return super().is_recording()

    def set_status(self, status: ot_span.Status | ot_span.StatusCode, description: str | None = None) -> None:
        return super().set_status(status, description)
    
    def record_exception(self, exception: Exception, attributes: Mapping[str, str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]] | None = None) -> None:
        return super().record_exception(exception, attributes)
    
    # end ot_span.Span requirements

