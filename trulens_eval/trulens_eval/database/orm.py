from __future__ import annotations

from sqlite3 import Connection as SQLite3Connection
from typing import ClassVar, Dict, Type, TypeVar

from sqlalchemy import Column
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import VARCHAR
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase

from trulens_eval import schema
from trulens_eval.utils.json import json_str_of_obj

TYPE_JSON = Text
"""Database type for JSON fields."""

TYPE_TIMESTAMP = Float
"""Database type for timestamps."""

TYPE_ENUM = Text
"""Database type for enum fields."""

TYPE_ID = VARCHAR(256)
"""Database type for unique IDs."""


class BaseWithTablePrefix(DeclarativeBase):
    """ORM base class except with __tablename__ defined in terms
    of a base name and a prefix.

    A subclass should set _table_base_name and/or _table_prefix. If it does not
    set both, make sure to set `__abstract__ = True`. Current design has
    subclasses set `_table_base_name` and then subclasses of that subclass
    setting `_table_prefix` as in `make_orm_for_prefix`.
    """

    # https://stackoverflow.com/questions/38245145/how-to-set-common-prefix-for-all-tables-in-sqlalchemy
    # Needed for sqlaclhemy to prevent it from creating a table for this class
    # before the two following attributes are set which we do in subclasses later.
    __abstract__ = True

    _table_base_name: str = "not set"
    """Base name for the table. 
    
    Will be prefixed by the prefix to create table names. This should be set by
    subclasses.
    """

    _table_prefix: str = ""
    """Prefix for the table name.
    
    This should be set by subclasses of subclasses of this class.
    """

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls._table_prefix + cls._table_base_name

class ORM():
    """Container for ORM classes. 
    
    Needs to be extended with classes that set table prefix.

    Warning:
        The relationships between tables established in the classes in this
        container refer to class names i.e. "AppDefinition" hence these are
        important and need to stay consistent between definition of one and
        relationships in another.
    """

    registry: Dict[str, Type[BaseWithTablePrefix]] = \
        BaseWithTablePrefix.registry._class_registry
    """Table name to ORM class mapping.
    
    This can be used to iterate through all classes/tables.
    """

    class AppDefinition(BaseWithTablePrefix):
        """ORM class for [AppDefinition][trulens_eval.schema.AppDefinition].

        Warning:
            We don't use any of the typical ORM features and this class is only
            used as a schema to interact with database through SQLAlchemy.
        """

        __abstract__ = True

        _table_base_name: ClassVar[str] = "apps"

        app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
        app_json = Column(TYPE_JSON, nullable=False)

        @classmethod
        def parse(
            cls,
            obj: schema.AppDefinition,
            redact_keys: bool = False
        ) -> ORM.AppDefinition:
            return cls(
                app_id=obj.app_id,
                app_json=obj.model_dump_json(redact_keys=redact_keys)
            )

    class FeedbackDefinition(BaseWithTablePrefix):
        """ORM class for [AppDefinition][trulens_eval.schema.FeedbackDefinition].

        Warning:
            We don't use any of the typical ORM features and this class is only
            used as a schema to interact with database through SQLAlchemy.
        """

        __abstract__ = True

        _table_base_name = "feedback_defs"

        feedback_definition_id = Column(
            TYPE_ID, nullable=False, primary_key=True
        )
        feedback_json = Column(TYPE_JSON, nullable=False)

        @classmethod
        def parse(
            cls,
            obj: schema.FeedbackDefinition,
            redact_keys: bool = False
        ) -> ORM.FeedbackDefinition:
            return cls(
                feedback_definition_id=obj.feedback_definition_id,
                feedback_json=json_str_of_obj(obj, redact_keys=redact_keys)
            )

    class Record(BaseWithTablePrefix):
        """ORM class for [AppDefinition][trulens_eval.schema.Record].

        Warning:
            We don't use any of the typical ORM features and this class is only
            used as a schema to interact with database through SQLAlchemy.
        """

        __abstract__ = True

        _table_base_name = "records"

        record_id = Column(TYPE_ID, nullable=False, primary_key=True)
        app_id = Column(TYPE_ID, nullable=False)
        input = Column(Text)
        output = Column(Text)
        record_json = Column(TYPE_JSON, nullable=False)
        tags = Column(Text, nullable=False)
        ts = Column(TYPE_TIMESTAMP, nullable=False)
        cost_json = Column(TYPE_JSON, nullable=False)
        perf_json = Column(TYPE_JSON, nullable=False)

        @declared_attr
        @classmethod
        def app(cls):
            return relationship(
                "AppDefinition",
                backref=backref('records', cascade="all,delete"),
                primaryjoin='AppDefinition.app_id == Record.app_id',
                foreign_keys=[cls.app_id],
            )

        @classmethod
        def parse(cls, obj: schema.Record, redact_keys: bool = False) -> ORM.Record:
            return cls(
                record_id=obj.record_id,
                app_id=obj.app_id,
                input=json_str_of_obj(obj.main_input, redact_keys=redact_keys),
                output=json_str_of_obj(obj.main_output, redact_keys=redact_keys),
                record_json=json_str_of_obj(obj, redact_keys=redact_keys),
                tags=obj.tags,
                ts=obj.ts.timestamp(),
                cost_json=json_str_of_obj(obj.cost, redact_keys=redact_keys),
                perf_json=json_str_of_obj(obj.perf, redact_keys=redact_keys),
            )

    class FeedbackResult(BaseWithTablePrefix):
        """
        ORM class for [AppDefinition][trulens_eval.schema.FeedbackResult].

        Warning:
            We don't use any of the typical ORM features and this class is only
            used as a schema to interact with database through SQLAlchemy.
        """

        __abstract__ = True

        _table_base_name = "feedbacks"

        feedback_result_id = Column(TYPE_ID, nullable=False, primary_key=True)
        record_id = Column(TYPE_ID, nullable=False)
        feedback_definition_id = Column(TYPE_ID, nullable=True)
        last_ts = Column(TYPE_TIMESTAMP, nullable=False)
        status = Column(TYPE_ENUM, nullable=False)
        error = Column(Text)
        calls_json = Column(TYPE_JSON, nullable=False)
        result = Column(Float)
        name = Column(Text, nullable=False)
        cost_json = Column(TYPE_JSON, nullable=False)
        multi_result = Column(TYPE_JSON)

        @declared_attr
        @classmethod
        def record(cls):
            return relationship(
                "Record",
                backref=backref('feedback_results', cascade="all,delete"),
                primaryjoin='Record.record_id == FeedbackResult.record_id',
                foreign_keys=[cls.record_id]
            )

        @declared_attr
        @classmethod
        def feedback_definition(cls):
            return relationship(
                "FeedbackDefinition",
                backref=backref("feedback_results", cascade="all,delete"),
                primaryjoin=
                "FeedbackDefinition.feedback_definition_id == FeedbackResult.feedback_definition_id",
                foreign_keys=[cls.feedback_definition_id],
            )

        @classmethod
        def parse(
            cls,
            obj: schema.FeedbackResult,
            redact_keys: bool = False
        ) -> ORM.FeedbackResult:
            return cls(
                feedback_result_id=obj.feedback_result_id,
                record_id=obj.record_id,
                feedback_definition_id=obj.feedback_definition_id,
                last_ts=obj.last_ts.timestamp(),
                status=obj.status.value,
                error=obj.error,
                calls_json=json_str_of_obj(
                    dict(calls=obj.calls), redact_keys=redact_keys
                ),
                result=obj.result,
                name=obj.name,
                cost_json=json_str_of_obj(obj.cost, redact_keys=redact_keys),
                multi_result=obj.multi_result
            )

T = TypeVar("T", bound=BaseWithTablePrefix)

def make_base_for_prefix(
    base_type: Type[T],
    table_prefix: str = "trulens_"
) -> Type[T]:
    """
    Create a base class for ORM classes with the given table name prefix.

    Args:
        base_type: Base class to extend. Should be a subclass of BaseWithTablePrefix.

        table_prefix: Prefix to use for table names.

    Returns:

        A class that extends `base_type` and sets the table prefix to `table_prefix`.
    """

    if not issubclass(base_type, BaseWithTablePrefix):
        raise ValueError("Expected `base_type` to be a subclass of `BaseWithTablePrefix`.")

    # sqlalchemy stores a mapping of class names to the classes we defined in
    # the ORM above. Here we want to create a class with the specific name
    # matching base_type hence use `type` instead of `class SomeName: ...`.
    return type(
        base_type.__name__, (base_type,), {"_table_prefix": table_prefix}
    )


def make_orm_for_prefix(table_prefix: str = "trulens_") -> Type[ORM]:
    """
    Make a container for ORM classes.

    This is done so that we can use a dynamic table name prefix and make the ORM
    classes based on that.

    Args:
        table_prefix: Prefix to use for table names.
    """

    class ORMWithPrefix(ORM):
        """ORM classes that have a table name prefix."""

        AppDefinition = make_base_for_prefix(ORM.AppDefinition, table_prefix)
        FeedbackDefinition = make_base_for_prefix(ORM.FeedbackDefinition, table_prefix)
        Record = make_base_for_prefix(ORM.Record, table_prefix)
        FeedbackResult = make_base_for_prefix(ORM.FeedbackResult, table_prefix)

    return ORMWithPrefix


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
