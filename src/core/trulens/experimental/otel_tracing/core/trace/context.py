# ruff: noqa: E402

""" """

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    Hashable,
    Optional,
    Union,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import span as span_api
import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils

if TYPE_CHECKING:
    from trulens.experimental.otel_tracing.core import trace as core_trace


logger = logging.getLogger(__name__)


class TraceState(serial_utils.SerialModel, span_api.TraceState):
    """[OTEL TraceState][opentelemetry.trace.TraceState] requirements.

    Adds [SerialModel][trulens.core.utils.serial.SerialModel] and therefore
    [pydantic.BaseModel][pydantic.BaseModel] onto the OTEL TraceState.
    """

    # Hackish: span_api.TraceState uses _dict internally.
    _dict: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)


class SpanContext(serial_utils.SerialModel, Hashable):
    """[OTEL SpanContext][opentelemetry.trace.SpanContext] requirements.

    Adds [SerialModel][trulens.core.utils.serial.SerialModel] and therefore
    [pydantic.BaseModel][pydantic.BaseModel] onto the OTEL SpanContext.

    Also adds hashing, equality, conversion, and representation methods.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # needed for enums that do not inherit from str
    )

    def __str__(self):
        return f"{self.trace_id % 0xFF:02x}/{self.span_id % 0xFF:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.trace_id + self.span_id

    def __eq__(self, other: ContextLike):
        if other is None:
            return False

        return self.trace_id == other.trace_id and self.span_id == other.span_id

    trace_id: types_schema.TraceID.PY_TYPE
    """Unique identifier for the trace.

    Each root span has a unique trace id."""

    span_id: types_schema.SpanID.PY_TYPE
    """Identifier for the span.

    Meant to be at least unique within the same trace_id.
    """

    trace_flags: trace_api.TraceFlags = pydantic.Field(
        trace_api.DEFAULT_TRACE_OPTIONS
    )

    @pydantic.field_validator("trace_flags", mode="before")
    @classmethod
    def _validate_trace_flags(cls, v):
        """Validate trace flags.

        Pydantic does not seem to like classes that inherit from int without this.
        """
        return trace_api.TraceFlags(v)

    trace_state: TraceState = pydantic.Field(default_factory=TraceState)

    is_remote: bool = False

    _tracer: core_trace.Tracer = pydantic.PrivateAttr(None)
    """Reference to the tracer that produces this SpanContext."""

    @property
    def tracer(self) -> core_trace.Tracer:
        """Tracer that produced this SpanContext."""
        return self._tracer

    @staticmethod
    def of_contextlike(
        context: ContextLike, tracer: Optional[core_trace.Tracer] = None
    ) -> SpanContext:
        """Convert several types that convey span/contxt identifiers into the
        common SpanContext type."""

        if isinstance(context, SpanContext):
            if tracer is not None:
                context._tracer = tracer

            return context

        if isinstance(context, span_api.SpanContext):
            # otel api SpanContext; doesn't have hashing and other things we need.
            return SpanContext(
                trace_id=context.trace_id,
                span_id=context.span_id,
                is_remote=context.is_remote,
                _tracer=tracer,
            )
        if isinstance(context, context_api.Context):
            # Context dict from OTEL.

            if len(context) == 1:
                span_encoding = next(iter(context.values()))

                return SpanContext(
                    trace_id=types_schema.TraceID.py_of_otel(
                        span_encoding.trace_id
                    ),
                    span_id=types_schema.SpanID.py_of_otel(
                        span_encoding.span_id
                    ),
                    _tracer=tracer,
                )
            else:
                raise ValueError(
                    f"Unrecognized context dict from OTEL: {context}"
                )
        if isinstance(context, dict):
            # Json encoding of SpanContext, i.e. output of
            # SpanContext.model_dump .

            context["_tracer"] = tracer
            return SpanContext.model_validate(context)

        raise ValueError(f"Unrecognized span context type: {context}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if v is None:
                continue
            # pydantic does not set private attributes in init
            if k.startswith("_") and hasattr(self, k):
                setattr(self, k, v)


ContextLike = Union[
    SpanContext, span_api.SpanContext, context_api.Context, serial_utils.JSON
]
"""SpanContext types we need to deal with.

These may be the non-hashable ones coming from OTEL, the hashable ones we
defined above, or their JSON representations."""
