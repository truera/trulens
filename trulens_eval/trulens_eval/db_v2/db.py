import logging
import warnings
from typing import List, Tuple, Sequence

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from trulens_eval import schema
from trulens_eval.db import DB
from trulens_eval.db_v2 import models
from trulens_eval.db_v2.migrations import migrate_db
from trulens_eval.db_v2.utils import for_all_methods, run_before, is_legacy_sqlite, is_memory_sqlite, \
    check_db_revision, migrate_legacy_sqlite

logger = logging.getLogger(__name__)


@for_all_methods(
    run_before(lambda self, *args, **kwargs: check_db_revision(self.engine)),
    _except=["migrate_database"]
)
class SqlAlchemyDB(DB):
    engine: Engine
    Session: sessionmaker

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, engine: Engine, **kwargs):
        kwargs["engine"] = engine
        kwargs["Session"] = sessionmaker(engine)
        super().__init__(**kwargs)

        if is_memory_sqlite(self.engine):
            warnings.warn(UserWarning(
                "SQLite in-memory may not be threadsafe. "
                "See https://www.sqlite.org/threadsafe.html"
            ))

    def migrate_database(self):
        """Migrate database schema to the latest revision"""
        if is_legacy_sqlite(self.engine):
            migrate_legacy_sqlite(self.engine)
        else:
            migrate_db(self.engine, revision="head")

    @classmethod
    def from_db_url(cls, url: str, **kwargs) -> "SqlAlchemyDB":
        return cls(engine=create_engine(url, **kwargs))

    def reset_database(self):
        raise NotImplementedError(
            f"Resetting the database is not implemented for `{self.__class__}`. "
            "Please perform this operation by connecting to the database directly"
        )

    def insert_record(self, record: schema.Record) -> schema.RecordID:
        pass  # TODO: impl

    def insert_app(self, app: schema.AppDefinition) -> schema.AppID:
        with self.Session.begin() as session:
            if _app := session.query(models.AppDefinition).filter_by(app_id=app.app_id).first():
                _app.app_json = app.json()
            else:
                _app = models.AppDefinition.parse(app)
                session.add(_app)
            return _app.app_id

    def insert_feedback_definition(self, feedback_definition: schema.FeedbackDefinition) -> schema.FeedbackDefinitionID:
        with self.Session.begin() as session:
            if _fb_def := session.query(models.FeedbackDefinition) \
                    .filter_by(feedback_definition_id=feedback_definition.feedback_definition_id) \
                    .first():
                _fb_def.app_json = feedback_definition.json()
            else:
                _fb_def = models.FeedbackDefinition.parse(feedback_definition)
                session.add(_fb_def)
            return _fb_def.app_id

    def insert_feedback(self, feedback_result: schema.FeedbackResult) -> schema.FeedbackResultID:
        pass  # TODO: impl

    def get_records_and_feedback(self, app_ids: List[str]) -> Tuple[pd.DataFrame, Sequence[str]]:
        pass  # TODO: impl
