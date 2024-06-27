from __future__ import annotations

import abc
import functools
from sqlite3 import Connection as SQLite3Connection
from typing import ClassVar, Dict, Generic, Type, TypeVar

from sqlalchemy import Column
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import VARCHAR
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref
from sqlalchemy.orm import configure_mappers
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import MetaData

from trulens_eval.database.base import DEFAULT_DATABASE_PREFIX
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.utils.json import json_str_of_obj

TYPE_JSON = Text
"""Database type for JSON fields."""

TYPE_TIMESTAMP = Float
"""Database type for timestamps."""

TYPE_ENUM = Text
"""Database type for enum fields."""

TYPE_ID = VARCHAR(256)
"""Database type for unique IDs."""


class BaseWithTablePrefix(
):  # to be mixed into DeclarativeBase or new_declarative_base()
    # Only for type hints or isinstance, issubclass checks.
    """ORM base class except with `__tablename__` defined in terms
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


T = TypeVar("T", bound=BaseWithTablePrefix)


# NOTE: lru_cache is important here as we don't want to create multiple classes
# of the same name for the same table name prefix as sqlalchemy will complain
# one some of our migration tools will not work.
@functools.lru_cache
def new_base(prefix: str) -> Type[T]:
    """Create a new base class for ORM classes.
    
    Note: This is a function to be able to define classes extending different
    SQLAlchemy delcarative bases. Each different such bases has a different set
    of mappings from classes to table names. If we only had one of these, our
    code will never be able to have two different sets of mappings at the same
    time. We need to be able to have multiple mappings for performing things
    such as database migrations and database copying from one database
    configuration to another.
    """

    base = declarative_base()
    return type(
        f"BaseWithTablePrefix{prefix}",
        (base, BaseWithTablePrefix),
        {
            "_table_prefix": prefix,
            "__abstract__":
                True  # stay abstract until _table_base_name is set in a subclass
        }
    )


class ORM(abc.ABC, Generic[T]):
    """Abstract definition of a container for ORM classes."""

    registry: Dict[str, Type[T]]
    metadata: MetaData

    AppDefinition: Type[T]
    FeedbackDefinition: Type[T]
    Record: Type[T]
    FeedbackResult: Type[T]


def new_orm(base: Type[T]) -> Type[ORM[T]]:
    """Create a new orm container from the given base table class."""

    class NewORM(ORM):
        """Container for ORM classes. 
        
        Needs to be extended with classes that set table prefix.

        Warning:
            The relationships between tables established in the classes in this
            container refer to class names i.e. "AppDefinition" hence these are
            important and need to stay consistent between definition of one and
            relationships in another.
        """

        registry: Dict[str, base] = \
            base.registry._class_registry
        """Table name to ORM class mapping for tables used by trulens_eval.
        
        This can be used to iterate through all classes/tables.
        """

        metadata: MetaData = base.metadata
        """SqlAlchemy metadata object for tables used by trulens_eval."""

        class AppDefinition(base):
            """ORM class for [AppDefinition][trulens_eval.schema.app.AppDefinition].

            Warning:
                We don't use any of the typical ORM features and this class is only
                used as a schema to interact with database through SQLAlchemy.
            """

            _table_base_name: ClassVar[str] = "apps"

            app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
            app_json = Column(TYPE_JSON, nullable=False)

            # records via one-to-many on Record.app_id
            # feedback_results via one-to-many on FeedbackResult.record_id

            @classmethod
            def parse(
                cls,
                obj: mod_app_schema.AppDefinition,
                redact_keys: bool = False
            ) -> ORM.AppDefinition:
                return cls(
                    app_id=obj.app_id,
                    app_json=obj.model_dump_json(redact_keys=redact_keys)
                )

        class FeedbackDefinition(base):
            """ORM class for [FeedbackDefinition][trulens_eval.schema.feedback.FeedbackDefinition].

            Warning:
                We don't use any of the typical ORM features and this class is only
                used as a schema to interact with database through SQLAlchemy.
            """

            _table_base_name = "feedback_defs"

            feedback_definition_id = Column(
                TYPE_ID, nullable=False, primary_key=True
            )
            feedback_json = Column(TYPE_JSON, nullable=False)

            # feedback_results via one-to-many on FeedbackResult.feedback_definition_id

            @classmethod
            def parse(
                cls,
                obj: mod_feedback_schema.FeedbackDefinition,
                redact_keys: bool = False
            ) -> ORM.FeedbackDefinition:
                return cls(
                    feedback_definition_id=obj.feedback_definition_id,
                    feedback_json=json_str_of_obj(obj, redact_keys=redact_keys)
                )

        class Record(base):
            """ORM class for [Record][trulens_eval.schema.record.Record].

            Warning:
                We don't use any of the typical ORM features and this class is only
                used as a schema to interact with database through SQLAlchemy.
            """

            _table_base_name = "records"

            record_id = Column(TYPE_ID, nullable=False, primary_key=True)
            app_id = Column(TYPE_ID, nullable=False)  # foreign key

            input = Column(Text)
            output = Column(Text)
            record_json = Column(TYPE_JSON, nullable=False)
            tags = Column(Text, nullable=False)
            ts = Column(TYPE_TIMESTAMP, nullable=False)
            cost_json = Column(TYPE_JSON, nullable=False)
            perf_json = Column(TYPE_JSON, nullable=False)

            app = relationship(
                'AppDefinition',
                backref=backref('records', cascade="all,delete"),
                primaryjoin='AppDefinition.app_id == Record.app_id',
                foreign_keys=app_id,
                order_by="(Record.ts,Record.record_id)"
            )

            @classmethod
            def parse(
                cls,
                obj: mod_record_schema.Record,
                redact_keys: bool = False
            ) -> ORM.Record:
                return cls(
                    record_id=obj.record_id,
                    app_id=obj.app_id,
                    input=json_str_of_obj(
                        obj.main_input, redact_keys=redact_keys
                    ),
                    output=json_str_of_obj(
                        obj.main_output, redact_keys=redact_keys
                    ),
                    record_json=json_str_of_obj(obj, redact_keys=redact_keys),
                    tags=obj.tags,
                    ts=obj.ts.timestamp(),
                    cost_json=json_str_of_obj(
                        obj.cost, redact_keys=redact_keys
                    ),
                    perf_json=json_str_of_obj(
                        obj.perf, redact_keys=redact_keys
                    ),
                )

        class FeedbackResult(base):
            """
            ORM class for [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult].

            Warning:
                We don't use any of the typical ORM features and this class is only
                used as a schema to interact with database through SQLAlchemy.
            """

            _table_base_name = "feedbacks"

            feedback_result_id = Column(
                TYPE_ID, nullable=False, primary_key=True
            )
            record_id = Column(TYPE_ID, nullable=False)  # foreign key
            feedback_definition_id = Column(
                TYPE_ID, nullable=False
            )  # foreign key
            last_ts = Column(TYPE_TIMESTAMP, nullable=False)
            status = Column(TYPE_ENUM, nullable=False)
            error = Column(Text)
            calls_json = Column(TYPE_JSON, nullable=False)
            result = Column(Float)
            name = Column(Text, nullable=False)
            cost_json = Column(TYPE_JSON, nullable=False)
            multi_result = Column(TYPE_JSON)

            record = relationship(
                'Record',
                backref=backref('feedback_results', cascade="all,delete"),
                primaryjoin='Record.record_id == FeedbackResult.record_id',
                foreign_keys=record_id,
                order_by=
                "(FeedbackResult.last_ts,FeedbackResult.feedback_result_id)"
            )

            feedback_definition = relationship(
                "FeedbackDefinition",
                backref=backref("feedback_results", cascade="all,delete"),
                primaryjoin=
                "FeedbackDefinition.feedback_definition_id == FeedbackResult.feedback_definition_id",
                foreign_keys=feedback_definition_id,
                order_by=
                "(FeedbackResult.last_ts,FeedbackResult.feedback_result_id)"
            )

            @classmethod
            def parse(
                cls,
                obj: mod_feedback_schema.FeedbackResult,
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
                    cost_json=json_str_of_obj(
                        obj.cost, redact_keys=redact_keys
                    ),
                    multi_result=obj.multi_result
                )

    configure_mappers()  # IMPORTANT
    # Without the above, orm class attributes which are defined using backref
    # will not be visible, i.e. orm.AppDefinition.records.

    # base.registry.configure()

    return NewORM


# NOTE: lru_cache is important here as we don't want to create multiple classes for
# the same table name as sqlalchemy will complain.
@functools.lru_cache
def make_base_for_prefix(
    base: Type[T],
    table_prefix: str = DEFAULT_DATABASE_PREFIX,
) -> Type[T]:
    """
    Create a base class for ORM classes with the given table name prefix.

    Args:
        base: Base class to extend. Should be a subclass of
            [BaseWithTablePrefix][trulens_eval.database.orm.BaseWithTablePrefix].

        table_prefix: Prefix to use for table names.

    Returns:
        A class that extends `base_type` and sets the table prefix to `table_prefix`.
    """

    if not hasattr(base, "_table_base_name"):
        raise ValueError(
            "Expected `base` to be a subclass of `BaseWithTablePrefix`."
        )

    # sqlalchemy stores a mapping of class names to the classes we defined in
    # the ORM above. Here we want to create a class with the specific name
    # matching base_type hence use `type` instead of `class SomeName: ...`.
    return type(base.__name__, (base,), {"_table_prefix": table_prefix})


# NOTE: lru_cache is important here as we don't want to create multiple classes for
# the same table name as sqlalchemy will complain.
@functools.lru_cache
def make_orm_for_prefix(
    table_prefix: str = DEFAULT_DATABASE_PREFIX
) -> Type[ORM[T]]:
    """
    Make a container for ORM classes.

    This is done so that we can use a dynamic table name prefix and make the ORM
    classes based on that.

    Args:
        table_prefix: Prefix to use for table names.
    """

    base: Type[T] = new_base(prefix=table_prefix)

    return new_orm(base)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
