import json
import logging
import warnings
from typing import List, Tuple, Sequence, Optional

import pandas as pd
from pydantic import Field
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from trulens_eval import schema
from trulens_eval.db import DB
from trulens_eval.db_v2 import orm
from trulens_eval.db_v2.migrations import migrate_db
from trulens_eval.db_v2.utils import for_all_methods, run_before, is_legacy_sqlite, is_memory_sqlite, \
    check_db_revision, migrate_legacy_sqlite
from trulens_eval.util import JSON

logger = logging.getLogger(__name__)


@for_all_methods(
    run_before(lambda self, *args, **kwargs: check_db_revision(self.engine)),
    _except=["migrate_database", "reload_engine"]
)
class SqlAlchemyDB(DB):
    engine_params: dict = Field(default_factory=dict)
    session_params: dict = Field(default_factory=dict)
    engine: Engine = None
    Session: sessionmaker = None

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reload_engine()
        if is_memory_sqlite(self.engine):
            warnings.warn(UserWarning(
                "SQLite in-memory may not be threadsafe. "
                "See https://www.sqlite.org/threadsafe.html"
            ))

    def reload_engine(self):
        self.engine = create_engine(**self.engine_params)
        self.Session = sessionmaker(self.engine, **self.session_params)

    @classmethod
    def from_db_url(cls, url: str) -> "SqlAlchemyDB":
        return cls(engine_params={"url": url})

    def migrate_database(self):
        """Migrate database schema to the latest revision"""
        if is_legacy_sqlite(self.engine):
            migrate_legacy_sqlite(self.engine)
        else:
            migrate_db(self.engine, revision="head")
        self.reload_engine()  # let sqlalchemy recognize the migrated schema

    def reset_database(self):
        raise NotImplementedError(
            f"Resetting the database is not implemented for `{self.__class__}`. "
            "Please perform this operation by connecting to the database directly"
        )

    def insert_record(self, record: schema.Record) -> schema.RecordID:
        with self.Session.begin() as session:
            _record = orm.Record.parse(record)
            session.add(_record)
            return _record.record_id

    def get_app(self, app_id: str) -> Optional[JSON]:
        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition).filter_by(app_id=app_id).first():
                return json.loads(_app.app_json)

    def insert_app(self, app: schema.AppDefinition) -> schema.AppID:
        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition).filter_by(app_id=app.app_id).first():
                _app.app_json = app.json()
            else:
                _app = orm.AppDefinition.parse(app)
                session.add(_app)
            return _app.app_id

    def insert_feedback_definition(self, feedback_definition: schema.FeedbackDefinition) -> schema.FeedbackDefinitionID:
        with self.Session.begin() as session:
            if _fb_def := session.query(orm.FeedbackDefinition) \
                    .filter_by(feedback_definition_id=feedback_definition.feedback_definition_id) \
                    .first():
                _fb_def.app_json = feedback_definition.json()
            else:
                _fb_def = orm.FeedbackDefinition.parse(feedback_definition)
                session.add(_fb_def)
            return _fb_def.feedback_definition_id

    def insert_feedback(self, feedback_result: schema.FeedbackResult) -> schema.FeedbackResultID:
        with self.Session.begin() as session:
            _feedback_result = orm.FeedbackResult.parse(feedback_result)
            session.add(_feedback_result)
            return _feedback_result.feedback_result_id

    def get_records_and_feedback(self, app_ids: List[str]) -> Tuple[pd.DataFrame, Sequence[str]]:
        # TODO: impl this
        pass
        # with self.Session.begin() as session:
        #     df = pd.DataFrame([], columns=[])
        #     for _record in session.query(orm.Record).filter(orm.Record.app_id.in_(app_ids)):
        #         _record.feedback_results
