from sqlite3 import Connection as SQLite3Connection

from sqlalchemy import Column, Text, VARCHAR, ForeignKey, Float, event, Engine
from sqlalchemy.orm import declarative_base, relationship

from trulens_eval import schema
from trulens_eval.util import json_str_of_obj

Base = declarative_base()

TYPE_JSON = Text
TYPE_TIMESTAMP = Float
TYPE_ENUM = Text


class AppDefinition(Base):
    __tablename__ = "apps"

    app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_json = Column(TYPE_JSON, nullable=False)

    records = relationship("Record", cascade="all,delete")

    @classmethod
    def parse(cls, obj: schema.AppDefinition) -> "AppDefinition":
        return cls(app_id=obj.app_id, app_json=obj.json())


class FeedbackDefinition(Base):
    __tablename__ = "feedback_defs"

    feedback_definition_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    feedback_json = Column(TYPE_JSON, nullable=False)

    feedback_results = relationship("FeedbackResult", cascade="all,delete")

    @classmethod
    def parse(cls, obj: schema.FeedbackDefinition) -> "FeedbackDefinition":
        return cls(feedback_definition_id=obj.feedback_definition_id, feedback_json=obj.json())


class Record(Base):
    __tablename__ = "records"

    record_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_id = Column(VARCHAR(256), ForeignKey("apps.app_id"), nullable=False)
    input = Column(Text)
    output = Column(Text)
    record_json = Column(TYPE_JSON, nullable=False)
    tags = Column(Text, nullable=False)
    ts = Column(TYPE_TIMESTAMP, nullable=False)
    cost_json = Column(TYPE_JSON, nullable=False)
    perf_json = Column(TYPE_JSON, nullable=False)

    app = relationship("AppDefinition", back_populates="records")
    feedback_results = relationship("FeedbackResult", cascade="all,delete")

    @classmethod
    def parse(cls, obj: schema.Record) -> "Record":
        return cls(
            record_id=obj.record_id,
            app_id=obj.app_id,
            input=json_str_of_obj(obj.main_input),
            output=json_str_of_obj(obj.main_output),
            record_json=json_str_of_obj(obj),
            tags=obj.tags,
            ts=obj.ts.timestamp(),
            cost_json=json_str_of_obj(obj.cost),
            perf_json=json_str_of_obj(obj.perf),
        )


class FeedbackResult(Base):
    __tablename__ = "feedbacks"

    feedback_result_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    record_id = Column(VARCHAR(256), ForeignKey("records.record_id"), nullable=False)
    feedback_definition_id = Column(VARCHAR(256), ForeignKey("feedback_defs.feedback_definition_id"), nullable=False)
    last_ts = Column(TYPE_TIMESTAMP, nullable=False)
    status = Column(TYPE_ENUM, nullable=False)
    error = Column(Text)
    calls_json = Column(TYPE_JSON, nullable=False)
    result = Column(Float)
    name = Column(Text, nullable=False)
    cost_json = Column(TYPE_JSON, nullable=False)
    multi_result = Column(TYPE_JSON)

    record = relationship("Record", back_populates="feedback_results")
    feedback_definition = relationship("FeedbackDefinition", back_populates="feedback_results")

    @classmethod
    def parse(cls, obj: schema.FeedbackResult) -> "FeedbackResult":
        return cls(
            feedback_result_id=obj.feedback_result_id,
            record_id=obj.record_id,
            feedback_definition_id=obj.feedback_definition_id,
            last_ts=obj.last_ts.timestamp(),
            status=obj.status.value,
            error=obj.error,
            calls_json=json_str_of_obj(dict(calls=obj.calls)),
            result=obj.result,
            name=obj.name,
            cost_json=json_str_of_obj(obj.cost),
            multi_result=obj.multi_result
        )


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
