from __future__ import annotations

import abc
from enum import Enum
from typing import Dict, Generic, Type, TypeVar

from sqlalchemy import BINARY
from sqlalchemy import DATETIME
from sqlalchemy import VARCHAR
from sqlalchemy import VARIANT
from sqlalchemy import Column
from sqlalchemy.orm import configure_mappers
from sqlalchemy.schema import MetaData
from trulens.core.database import orm as db_orm

T = TypeVar("T", bound=db_orm.BaseWithTablePrefix)


class SpanORM(abc.ABC, Generic[T]):
    """Abstract definition of a container for ORM classes."""

    registry: Dict[str, Type[T]]
    metadata: MetaData

    Span: Type[T]


class SpanType(str, Enum):
    TRACE_ROOT = "trace"

    EVAL_ROOT = "eval"

    CONTEXT_ROOT = "context"


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

            # OTEL requirements that we use:
            span_id: Column = Column(
                BINARY(64), nullable=False, primary_key=True
            )
            trace_id: Column = Column(
                BINARY(128), nullable=False, primary_key=True
            )
            parent_span_id: Column = Column(BINARY(64))
            parent_trace_id: Column = Column(BINARY(128))

            name: Column = Column(VARCHAR(32), nullable=False)
            start_time: Column = Column(DATETIME, nullable=False)
            end_time: Column = Column(DATETIME, nullable=False)
            attributes: Column = Column(VARIANT)
            span_kind: Column = Column(VARCHAR(32), nullable=False)
            status: Column = Column(VARCHAR(32), nullable=False)

            # Note that there are other OTEL requirements that we do not use and
            # hence do not model here.

            # Other columns:
            span_type: Column = Column(VARCHAR(256), nullable=False)

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible, i.e. orm.AppDefinition.records.

    # base.registry.configure()

    return NewSpanORM
