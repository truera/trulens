from sqlite3 import Connection as SQLite3Connection

from sqlalchemy import Column
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import VARCHAR
from sqlalchemy.orm import backref
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

from trulens_eval import schema
from trulens_eval.utils.json import json_str_of_obj

Base = declarative_base()

TYPE_JSON = Text
TYPE_TIMESTAMP = Float
TYPE_ENUM = Text


class AppDefinition(Base):
    __tablename__ = "apps"

    app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_json = Column(TYPE_JSON, nullable=False)

    @classmethod
    def parse(
        cls,
        obj: schema.AppDefinition,
        redact_keys: bool = False
    ) -> "AppDefinition":
        return cls(
            app_id=obj.app_id,
            app_json=obj.model_dump_json(redact_keys=redact_keys)
        )


class FeedbackDefinition(Base):
    __tablename__ = "feedback_defs"

    feedback_definition_id = Column(
        VARCHAR(256), nullable=False, primary_key=True
    )
    feedback_json = Column(TYPE_JSON, nullable=False)

    @classmethod
    def parse(
        cls,
        obj: schema.FeedbackDefinition,
        redact_keys: bool = False
    ) -> "FeedbackDefinition":
        return cls(
            feedback_definition_id=obj.feedback_definition_id,
            feedback_json=json_str_of_obj(obj, redact_keys=redact_keys)
        )


class Record(Base):
    __tablename__ = "records"

    record_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_id = Column(VARCHAR(256), nullable=False)
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
        foreign_keys=[app_id],
    )

    @classmethod
    def parse(cls, obj: schema.Record, redact_keys: bool = False) -> "Record":
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


class FeedbackResult(Base):
    __tablename__ = "feedbacks"

    feedback_result_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    record_id = Column(VARCHAR(256), nullable=False)
    feedback_definition_id = Column(VARCHAR(256), nullable=True)
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
        foreign_keys=[record_id]
    )

    feedback_definition = relationship(
        "FeedbackDefinition",
        backref=backref("feedback_results", cascade="all,delete"),
        primaryjoin=
        "FeedbackDefinition.feedback_definition_id == FeedbackResult.feedback_definition_id",
        foreign_keys=[feedback_definition_id],
    )

    @classmethod
    def parse(
        cls,
        obj: schema.FeedbackResult,
        redact_keys: bool = False
    ) -> "FeedbackResult":
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


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
