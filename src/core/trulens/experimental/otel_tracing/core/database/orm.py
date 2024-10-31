from __future__ import annotations

import abc
from typing import ClassVar, Dict, Generic, Type, TypeVar

from sqlalchemy import BINARY
from sqlalchemy import INTEGER
from sqlalchemy import JSON
from sqlalchemy import SMALLINT
from sqlalchemy import TIMESTAMP
from sqlalchemy import VARCHAR
from sqlalchemy import Column
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import configure_mappers
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import func
from trulens.core._utils.pycompat import TypeAlias  # import style exception
from trulens.core.database import orm as db_orm
from trulens.core.utils import containers as container_utils
from trulens.experimental.otel_tracing.core import otel as core_otel
from trulens.experimental.otel_tracing.core import sem as core_sem
from trulens.experimental.otel_tracing.core import trace as core_trace

T = TypeVar("T", bound=db_orm.BaseWithTablePrefix)


TSpanID: TypeAlias = int
"""Type of span identifiers.

64 bit int as per OpenTelemetry. Viewed as int in python and BINARY in the db.
"""
NUM_SPANID_BITS: int = 64
"""OTEL: Number of bits in a span identifier."""

TTraceID: TypeAlias = int
"""Type of trace identifiers.

128 bit int as per OpenTelemetry. Viewed as int in python and BINARY in the db.
"""
NUM_TRACEID_BITS: int = 128
"""OTEL: Number of bits in a trace identifier."""

TTimestamp: TypeAlias = int
"""Type of timestamps in spans.

64 bit int representing nanoseconds since epoch as per OpenTelemetry. Viewed as
int in python and TIMESTAMP in the db.
"""
NUM_TIMESTAMP_BITS = 64
"""OTEL: Number of bits in a timestamp."""

NUM_SPANTYPE_BYTES = 32
# trulens specific, not otel

NUM_NAME_BYTES = 32
NUM_STATUS_DESCRIPTION_BYTES = 256
# TODO: match otel spec


class SpanORM(abc.ABC, Generic[T]):
    """Abstract definition of a container for ORM classes."""

    registry: Dict[str, Type[T]]
    metadata: MetaData

    Span: Type[T]


def new_orm(base: Type[T], prefix: str = "trulens_") -> Type[SpanORM[T]]:
    """Create a new orm container from the given base table class."""

    class NewSpanORM(SpanORM):
        """Container for ORM classes.

        Needs to be extended with classes that set table prefix.

        Warning:
            The relationships between tables established in the classes in this
            container refer to class names i.e. "AppDefinition" hence these are
            important and need to stay consistent between definition of one and
            relationships in another.
        """

        registry: Dict[str, base] = base.registry._class_registry
        """Table name to ORM class mapping for tables used by trulens.

        This can be used to iterate through all classes/tables.
        """

        metadata: MetaData = base.metadata
        """SqlAlchemy metadata object for tables used by trulens."""

        class Span(base):
            """Span DB schema."""

            _table_base_name: ClassVar[str] = "spans"

            # pagination utility columns
            created_timestamp: Column = Column(
                TIMESTAMP,
                server_default=func.now(),
            )
            updated_timestamp: Column = Column(
                TIMESTAMP,
                server_default=func.now(),
                onupdate=func.current_timestamp(),
            )

            index: Column = Column(
                INTEGER, primary_key=True, autoincrement=True
            )

            # OTEL requirements that we use:
            span_id: Column = Column(
                BINARY(NUM_SPANID_BITS // 8), nullable=False
            )
            trace_id: Column = Column(
                BINARY(NUM_TRACEID_BITS // 8), nullable=False
            )

            __table_args__ = (UniqueConstraint("span_id", "trace_id"),)

            parent_span_id: Column = Column(BINARY(NUM_SPANID_BITS // 8))
            parent_trace_id: Column = Column(BINARY(NUM_TRACEID_BITS // 8))

            name: Column = Column(VARCHAR(NUM_NAME_BYTES), nullable=False)
            start_timestamp: Column = Column(TIMESTAMP, nullable=False)
            end_timestamp: Column = Column(TIMESTAMP, nullable=False)
            attributes: Column = Column(JSON)
            kind: Column = Column(SMALLINT, nullable=False)
            status: Column = Column(SMALLINT, nullable=False)
            status_description: Column = Column(
                VARCHAR(NUM_STATUS_DESCRIPTION_BYTES)
            )

            # Note that there are other OTEL requirements that we do not use and
            # hence do not model here.

            # Other columns:
            span_type: Column = Column(
                VARCHAR(NUM_SPANTYPE_BYTES), nullable=False
            )

            @classmethod
            def parse(cls, obj: core_otel.Span) -> NewSpanORM.Span:
                """Parse a span object into an ORM object."""

                return cls(
                    span_id=obj.context.span_id.to_bytes(NUM_SPANID_BITS // 8),
                    trace_id=obj.context.trace_id.to_bytes(
                        NUM_TRACEID_BITS // 8
                    ),
                    parent_span_id=obj.parent.span_id.to_bytes(
                        NUM_SPANID_BITS // 8
                    )
                    if obj.parent
                    else None,
                    parent_trace_id=obj.parent.trace_id.to_bytes(
                        NUM_TRACEID_BITS // 8
                    )
                    if obj.parent
                    else None,
                    name=obj.name,
                    start_timestamp=container_utils.datetime_of_ns_timestamp(
                        obj.start_timestamp
                    ),
                    end_timestamp=container_utils.datetime_of_ns_timestamp(
                        obj.end_timestamp
                    )
                    if obj.end_timestamp
                    else None,
                    attributes=obj.attributes,
                    kind=obj.kind,
                    status=obj.status,
                    status_description=obj.status_description,
                    span_type=obj.span_type
                    if hasattr(obj, "span_type")
                    else "unknown",
                )

            def write(self) -> core_trace.Span:
                """Convert ORM class to span"""

                context = core_trace.SpanContext(
                    trace_id=int.from_bytes(self.trace_id),
                    span_id=int.from_bytes(self.span_id),
                )

                parent = (
                    core_trace.SpanContext(
                        trace_id=int.from_bytes(self.parent_trace_id),
                        span_id=int.from_bytes(self.parent_span_id),
                    )
                    if self.parent_span_id
                    else None
                )

                return core_sem.TypedSpan(
                    name=self.name,
                    context=context,
                    parent=parent,
                    kind=self.kind,
                    attributes=self.attributes,
                    start_timestamp=container_utils.ns_timestamp_of_datetime(
                        self.start_timestamp
                    ),
                    end_timestamp=container_utils.ns_timestamp_of_datetime(
                        self.end_timestamp
                    )
                    if self.end_timestamp
                    else None,
                    status=self.status,
                    status_description=self.status_description,
                    links=[],  # we dont keep links
                )

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible.

    return NewSpanORM
