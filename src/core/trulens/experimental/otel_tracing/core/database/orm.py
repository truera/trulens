from __future__ import annotations

import abc
from typing import (
    ClassVar,
    Dict,
    Generic,
    Type,
    TypeVar,
)

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
from trulens.core.schema import types as types_schema
from trulens.experimental.otel_tracing.core import sem as core_sem
from trulens.experimental.otel_tracing.core import trace as core_trace

NUM_SPANTYPE_BYTES = 32
# trulens specific, not otel

NUM_NAME_BYTES = 32
NUM_STATUS_DESCRIPTION_BYTES = 256
# TODO: match otel spec

T = TypeVar("T")


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
                types_schema.SpanID.SQL_SCHEMA_TYPE, nullable=False
            )
            trace_id: Column = Column(
                types_schema.TraceID.SQL_SCHEMA_TYPE, nullable=False
            )

            __table_args__ = (UniqueConstraint("span_id", "trace_id"),)

            parent_span_id: Column = Column(types_schema.SpanID.SQL_SCHEMA_TYPE)
            parent_trace_id: Column = Column(
                types_schema.TraceID.SQL_SCHEMA_TYPE
            )

            name: Column = Column(VARCHAR(NUM_NAME_BYTES), nullable=False)
            start_timestamp: Column = Column(
                types_schema.Timestamp.SQL_SCHEMA_TYPE, nullable=False
            )
            end_timestamp: Column = Column(
                types_schema.Timestamp.SQL_SCHEMA_TYPE, nullable=False
            )
            attributes: Column = Column(JSON)
            kind: Column = Column(SMALLINT, nullable=False)  # better type?
            status: Column = Column(SMALLINT, nullable=False)  # better type?
            status_description: Column = Column(
                VARCHAR(NUM_STATUS_DESCRIPTION_BYTES)
            )

            # Note that there are other OTEL requirements that we do not use and
            # hence do not model here.

            # Other columns:
            record_ids: Column = Column(
                types_schema.TraceRecordIDs.SQL_SCHEMA_TYPE
            )
            """Each main app method call gets a different record_id.

            We cannot use trace_id for this purpose without interfering with
            expected OTEL behaviour. Can be null if we cannot figure out what
            record/call this span is associated with."""
            # TODO: figure out the constraints/uniqueness for record_ids.

            span_types: Column = Column(JSON, nullable=False)
            """Types (of TypedSpan) the span belongs to."""

            @classmethod
            def parse(cls, obj: core_sem.TypedSpan) -> NewSpanORM.Span:
                """Parse a span object into an ORM object."""

                assert isinstance(
                    obj, core_sem.TypedSpan
                ), "TypedSpan expected."

                return cls(
                    span_id=types_schema.SpanID.sql_of_py(obj.context.span_id),
                    trace_id=types_schema.TraceID.sql_of_py(
                        obj.context.trace_id
                    ),
                    parent_span_id=types_schema.SpanID.sql_of_py(
                        obj.parent.span_id
                    )
                    if obj.parent
                    else None,
                    parent_trace_id=types_schema.TraceID.sql_of_py(
                        obj.parent.trace_id
                    )
                    if obj.parent
                    else None,
                    name=obj.name,
                    start_timestamp=types_schema.Timestamp.sql_of_py(
                        obj.start_timestamp
                    ),
                    end_timestamp=types_schema.Timestamp.sql_of_py(
                        obj.end_timestamp
                    )
                    if obj.end_timestamp
                    else None,
                    attributes=obj.attributes,
                    kind=obj.kind.value,  # same as below
                    status=obj.status.value,  # doesn't like to take the Enum itself
                    status_description=obj.status_description,
                    span_types=list(t.value for t in obj.span_types),
                )

            def write(self) -> core_trace.Span:
                """Convert ORM class to span"""

                context = core_trace.SpanContext(
                    trace_id=types_schema.TraceID.py_of_sql(self.trace_id),
                    span_id=types_schema.SpanID.py_of_sql(self.span_id),
                )

                parent = (
                    core_trace.SpanContext(
                        trace_id=types_schema.TraceID.py_of_sql(
                            self.parent_trace_id
                        ),
                        span_id=types_schema.SpanID.py_of_sql(
                            self.parent_span_id
                        ),
                    )
                    if self.parent_span_id
                    else None
                )

                return core_sem.TypedSpan.mixin_new(
                    name=self.name,
                    context=context,
                    parent=parent,
                    kind=self.kind,
                    attributes=self.attributes,
                    start_timestamp=self.start_timestamp,
                    end_timestamp=self.end_timestamp,
                    status=self.status,
                    status_description=self.status_description,
                    links=[],  # we dont keep links
                    span_types=self.span_types,
                )

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible.

    return NewSpanORM
