from __future__ import annotations

import abc
from typing import Dict, Generic, Type, TypeVar

from sqlalchemy import BINARY
from sqlalchemy import INTEGER
from sqlalchemy import JSON
from sqlalchemy import TIMESTAMP
from sqlalchemy import VARCHAR
from sqlalchemy import Column
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import configure_mappers
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import func
from trulens.core.database import orm as db_orm

T = TypeVar("T", bound=db_orm.BaseWithTablePrefix)


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
            """Base class for all ORM classes."""

            __table_base_name__ = "spans"

            # common attributes for all tables:
            ts: Column = Column(
                TIMESTAMP,
                server_default=func.now(),
                onupdate=func.current_timestamp(),
            )
            id: Column = Column(INTEGER, primary_key=True, autoincrement=True)

            # OTEL requirements that we use:
            span_id: Column = Column(
                BINARY(64), nullable=False, primary_key=True
            )
            trace_id: Column = Column(
                BINARY(128), nullable=False, primary_key=True
            )

            __table_args__ = (
                UniqueConstraint("span_id", "trace_id", name="span_trace_ids"),
            )

            parent_span_id: Column = Column(BINARY(64))
            parent_trace_id: Column = Column(BINARY(128))

            name: Column = Column(VARCHAR(32), nullable=False)
            start_timestamp: Column = Column(TIMESTAMP, nullable=False)
            end_timestamp: Column = Column(TIMESTAMP, nullable=False)
            attributes: Column = Column(JSON)
            kind: Column = Column(VARCHAR(32), nullable=False)
            status: Column = Column(VARCHAR(32), nullable=False)
            status_description: Column = Column(VARCHAR(256))

            # Note that there are other OTEL requirements that we do not use and
            # hence do not model here.

            # Other columns:
            span_type: Column = Column(VARCHAR(256), nullable=False)

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible.

    return NewSpanORM
