import json
import logging
import warnings
from datetime import datetime
from typing import List, Tuple, Sequence, Optional, Iterable, Union

import numpy as np
import pandas as pd
from pydantic import Field
from sqlalchemy import Engine, create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from trulens_eval import schema
from trulens_eval.db import DB
from trulens_eval.db_v2 import orm
from trulens_eval.db_v2.migrations import upgrade_db
from trulens_eval.db_v2.utils import for_all_methods, run_before, is_legacy_sqlite, is_memory_sqlite, \
    check_db_revision, migrate_legacy_sqlite
from trulens_eval.schema import RecordID, FeedbackResultID, FeedbackDefinitionID, FeedbackResultStatus
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
            upgrade_db(self.engine, revision="head")
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

    def get_feedback(
            self,
            record_id: Optional[RecordID] = None,
            feedback_result_id: Optional[FeedbackResultID] = None,
            feedback_definition_id: Optional[FeedbackDefinitionID] = None,
            status: Optional[FeedbackResultStatus] = None,
            last_ts_before: Optional[datetime] = None
    ) -> pd.DataFrame:
        pass  # TODO

    def get_records_and_feedback(self, app_ids: List[str]) -> Tuple[pd.DataFrame, Sequence[str]]:
        with self.Session.begin() as session:
            stmt = select(orm.AppDefinition).where(orm.AppDefinition.app_id.in_(app_ids))
            apps = (row[0] for row in session.execute(stmt).all())
            return AppsExtractor().get_df_and_cols(apps)


class AppsExtractor:
    app_cols = ["app_id", "app_json"]
    rec_cols = ["record_id", "input", "output", "tags", "record_json", "cost_json", "perf_json", "ts"]

    def __init__(self):
        self.feedback_columns = set()

    def get_df_and_cols(self, apps: Iterable[orm.AppDefinition]) -> Tuple[pd.DataFrame, Sequence[str]]:
        df = pd.concat(self.extract_apps(apps))
        return df, list(self.feedback_columns)

    def extract_apps(self, apps: Iterable[orm.AppDefinition]) -> Iterable[pd.DataFrame]:
        yield pd.DataFrame([], columns=self.app_cols + self.rec_cols)  # prevent empty iterator
        for _app in apps:
            if _recs := _app.records:
                df = pd.concat(self.extract_records(_recs))

                for col in self.app_cols:
                    df[col] = getattr(_app, col)

                yield df

    def extract_records(self, records: Iterable[orm.Record]) -> Iterable[pd.DataFrame]:
        for _rec in records:
            df = pd.DataFrame(self.extract_results(_rec.feedback_results), columns=["key", "value"]) \
                .pivot_table(columns="key", values="value", aggfunc=self.agg_result_or_calls) \
                .reset_index(drop=True).rename_axis("", axis=1)

            for col in self.rec_cols:
                df[col] = datetime.fromtimestamp(_rec.ts).isoformat() if col == "ts" else getattr(_rec, col)

            yield df

    def extract_results(self, results: Iterable[orm.FeedbackResult]) -> Iterable[Tuple[str, Union[float, dict]]]:
        for _res in results:
            self.feedback_columns.add(_res.name)
            yield _res.name, _res.result
            yield f"{_res.name}_calls", json.loads(_res.calls_json)["calls"][0]

    @classmethod
    def agg_result_or_calls(cls, *args):
        if not args:
            return None
        if len(args) == 1:
            return args[0]
        if isinstance(args[0], dict):
            return args
        else:
            return np.mean(args)
