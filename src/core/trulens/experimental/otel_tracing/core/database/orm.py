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
from sqlalchemy import Column
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import configure_mappers
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import func
from trulens.core.schema import types as types_schema
from trulens.experimental.otel_tracing.core.trace import context as core_context
from trulens.experimental.otel_tracing.core.trace import sem as core_sem

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
                types_schema.Timestamp.SQL_SCHEMA_TYPE,
                server_default=func.now(),
            )
            updated_timestamp: Column = Column(
                types_schema.Timestamp.SQL_SCHEMA_TYPE,
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

            name: Column = Column(
                types_schema.SpanName.SQL_SCHEMA_TYPE, nullable=False
            )

            start_timestamp: Column = Column(
                types_schema.Timestamp.SQL_SCHEMA_TYPE, nullable=False
            )
            end_timestamp: Column = Column(
                types_schema.Timestamp.SQL_SCHEMA_TYPE, nullable=False
            )
            attributes: Column = Column(JSON)

            kind: Column = Column(
                types_schema.SpanKind.SQL_SCHEMA_TYPE, nullable=False
            )

            status: Column = Column(
                types_schema.SpanStatusCode.SQL_SCHEMA_TYPE, nullable=False
            )

            status_description: Column = Column(
                types_schema.StatusDescription.SQL_SCHEMA_TYPE
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

            span_types: Column = Column(JSON, nullable=False)
            """Types (of TypedSpan) the span belongs to."""

            @classmethod
            def parse(cls, obj: core_sem.TypedSpan) -> NewSpanORM.Span:
                """Parse a typed span object into an ORM object."""

                record_ids = {}
                if isinstance(obj, core_sem.Record):
                    if obj.record_ids is not None:
                        record_ids = obj.record_ids
                    else:
                        raise NotImplementedError(
                            f"Record spans must have record_ids. This span does not: {obj}"
                        )
                else:
                    # TODO: figure out how to handle this case, or just not include these?
                    raise NotImplementedError("Cannot handle non-App spans.")

                assert isinstance(
                    obj, core_sem.TypedSpan
                ), "TypedSpan expected."

                return cls(
                    span_id=types_schema.SpanID.sql_of_py(obj.context.span_id),
                    trace_id=types_schema.TraceID.sql_of_py(
                        obj.context.trace_id
                    ),
                    record_ids=types_schema.TraceRecordIDs.sql_of_py(
                        record_ids
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
                    kind=types_schema.SpanKind.sql_of_py(obj.kind),
                    status=types_schema.SpanStatusCode.sql_of_py(obj.status),
                    status_description=types_schema.StatusDescription.sql_of_py(
                        obj.status_description
                    )
                    if obj.status_description
                    else None,
                    span_types=list(t.value for t in obj.span_types),
                )

            def write(self) -> core_sem.TypedSpan:
                """Convert ORM class to typed span."""

                context = core_context.SpanContext(
                    trace_id=types_schema.TraceID.py_of_sql(self.trace_id),
                    span_id=types_schema.SpanID.py_of_sql(self.span_id),
                )

                parent = (
                    core_context.SpanContext(
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

                span_types = set(self.span_types)
                # other_args = {}
                # if truconv.SpanAttributes.SpanType.RECORD_ROOT in span_types:
                # TODO: need to recover AttributeProperty fields from attributes here.
                #    other_args["record_id"] = self.attributes[
                #        truconv.SpanAttributes.RECORD_ROOT.RECORD_ID
                #    ]

                return core_sem.TypedSpan.mixin_new(
                    name=self.name,
                    context=context,
                    parent=parent,
                    kind=types_schema.SpanKind.py_of_sql(self.kind),
                    attributes=self.attributes,
                    start_timestamp=types_schema.Timestamp.py_of_sql(
                        self.start_timestamp
                    ),
                    end_timestamp=types_schema.Timestamp.py_of_sql(
                        self.end_timestamp
                    ),
                    status=types_schema.SpanStatusCode.py_of_sql(self.status),
                    status_description=types_schema.StatusDescription.py_of_sql(
                        self.status_description
                    ),
                    links=[],  # we dont keep links
                    span_types=types_schema.SpanTypes.py_of_sql(span_types),
                    record_ids=types_schema.TraceRecordIDs.py_of_sql(
                        self.record_ids
                    ),
                    #    **other_args,
                )

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible.

    return NewSpanORM
