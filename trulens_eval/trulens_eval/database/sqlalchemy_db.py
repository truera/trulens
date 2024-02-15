from collections import defaultdict
from datetime import datetime
import json
import logging
from typing import (
    Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Union
)
import warnings

import numpy as np
import pandas as pd
from pydantic import Field
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

from trulens_eval import schema
from trulens_eval.database import orm
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.migrations.db_data_migration import data_migrate
from trulens_eval.database.orm import AppDefinition
from trulens_eval.database.orm import FeedbackDefinition
from trulens_eval.database.orm import FeedbackResult
from trulens_eval.database.orm import Record
from trulens_eval.database.utils import check_db_revision
from trulens_eval.database.utils import for_all_methods
from trulens_eval.database.utils import is_legacy_sqlite
from trulens_eval.database.utils import is_memory_sqlite
from trulens_eval.database.utils import migrate_legacy_sqlite
from trulens_eval.database.utils import run_before
from trulens_eval.db import DB
from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.schema import FeedbackDefinitionID
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import Perf
from trulens_eval.schema import RecordID
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK
from trulens_eval.utils.text import UNICODE_HOURGLASS
from trulens_eval.utils.text import UNICODE_STOP

logger = logging.getLogger(__name__)


@for_all_methods(
    run_before(lambda self, *args, **kwargs: check_db_revision(self.engine)),
    _except=[
        "migrate_database",
        "reload_engine",
        "reset_database"  # migrates database automatically
    ]
)
class SqlAlchemyDB(DB):
    engine_params: dict = Field(default_factory=dict)
    session_params: dict = Field(default_factory=dict)
    engine: Engine = None
    Session: sessionmaker = None

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def __init__(self, redact_keys: bool = False, **kwargs):
        super().__init__(redact_keys=redact_keys, **kwargs)
        self.reload_engine()
        if is_memory_sqlite(self.engine):
            warnings.warn(
                UserWarning(
                    "SQLite in-memory may not be threadsafe. "
                    "See https://www.sqlite.org/threadsafe.html"
                )
            )

    def reload_engine(self):
        self.engine = create_engine(**self.engine_params)
        self.Session = sessionmaker(self.engine, **self.session_params)

    @classmethod
    def from_db_url(cls, url: str, redact_keys: bool = False) -> "SqlAlchemyDB":
        # Params needed for https://github.com/truera/trulens/issues/470
        # Params are from
        # https://stackoverflow.com/questions/55457069/how-to-fix-operationalerror-psycopg2-operationalerror-server-closed-the-conn

        engine_params = {
            "url": url,
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
        }

        if not is_memory_sqlite(url=url):
            # These params cannot be given to memory-based sqlite engine.
            engine_params["max_overflow"] = 2
            engine_params["pool_use_lifo"] = True

        return cls(
            engine_params=engine_params,
            redact_keys=redact_keys
        )

    def migrate_database(self):
        """
        Migrate database schema to the latest revision.
        """

        try:
            # Expect to get the the behind exception.
            check_db_revision(self.engine)

        except DatabaseVersionException as e:
            if e.reason == DatabaseVersionException.Reason.BEHIND:
                revisions = DbRevisions.load(self.engine)
                from_version = revisions.current
                ### SCHEMA MIGRATION ###
                if is_legacy_sqlite(self.engine):
                    migrate_legacy_sqlite(self.engine)
                else:
                    ## TODO Create backups here. This is not sqlalchemy's strong suit: https://stackoverflow.com/questions/56990946/how-to-backup-up-a-sqlalchmey-database
                    ### We might allow migrate_database to take a backup url (and suggest user to supply if not supplied ala `tru.migrate_database(backup_db_url="...")`)
                    ### We might try _copy_database as a backup, but it would need to automatically handle clearing the db, and also current implementation requires migrate to run first.
                    ### A valid backup would need to be able to copy an old version, not the newest version
                    upgrade_db(self.engine, revision="head")

                self.reload_engine(
                )  # let sqlalchemy recognize the migrated schema

                ### DATA MIGRATION ###
                data_migrate(self, from_version)
                return

            elif e.reason == DatabaseVersionException.Reason.AHEAD:
                # Rethrow the ahead message suggesting to upgrade trulens_eval.
                raise e

            else:
                # TODO: better message here for unhandled cases?
                raise e

        # If we get here, our db revision does not need upgrade.
        logger.info("Your database does not need migration.")

    def reset_database(self):
        meta = MetaData()
        meta.reflect(bind=self.engine)
        meta.drop_all(bind=self.engine)

        self.migrate_database()

    def insert_record(self, record: schema.Record) -> schema.RecordID:
        # TODO: thread safety

        _rec = orm.Record.parse(record, redact_keys=self.redact_keys)
        with self.Session.begin() as session:
            if session.query(orm.Record).filter_by(record_id=record.record_id
                                                  ).first():
                session.merge(_rec)  # update existing
            else:
                session.merge(_rec)  # add new record # .add was not thread safe

            logger.info(f"{UNICODE_CHECK} added record {_rec.record_id}")

            return _rec.record_id

    def get_app(self, app_id: str) -> Optional[JSON]:
        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition).filter_by(app_id=app_id
                                                                 ).first():
                return json.loads(_app.app_json)

    def get_apps(self) -> Iterable[JSON]:
        with self.Session.begin() as session:
            for _app in session.query(orm.AppDefinition):
                yield json.loads(_app.app_json)

    def insert_app(self, app: schema.AppDefinition) -> schema.AppID:
        # TODO: thread safety

        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition
                                    ).filter_by(app_id=app.app_id).first():

                _app.app_json = app.model_dump_json()
            else:
                _app = orm.AppDefinition.parse(
                    app, redact_keys=self.redact_keys
                )
                session.merge(_app)  # .add was not thread safe

            logger.info(f"{UNICODE_CHECK} added app {_app.app_id}")

            return _app.app_id

    def insert_feedback_definition(
        self, feedback_definition: schema.FeedbackDefinition
    ) -> schema.FeedbackDefinitionID:
        # TODO: thread safety

        with self.Session.begin() as session:
            if _fb_def := session.query(orm.FeedbackDefinition) \
                    .filter_by(feedback_definition_id=feedback_definition.feedback_definition_id) \
                    .first():
                _fb_def.app_json = feedback_definition.model_dump_json()
            else:
                _fb_def = orm.FeedbackDefinition.parse(
                    feedback_definition, redact_keys=self.redact_keys
                )
                session.merge(_fb_def)  # .add was not thread safe

            logger.info(
                f"{UNICODE_CHECK} added feedback definition {_fb_def.feedback_definition_id}"
            )

            return _fb_def.feedback_definition_id

    def get_feedback_defs(
        self, feedback_definition_id: Optional[str] = None
    ) -> pd.DataFrame:
        with self.Session.begin() as session:
            q = select(orm.FeedbackDefinition)
            if feedback_definition_id:
                q = q.filter_by(feedback_definition_id=feedback_definition_id)
            fb_defs = (row[0] for row in session.execute(q))
            return pd.DataFrame(
                data=(
                    (fb.feedback_definition_id, json.loads(fb.feedback_json))
                    for fb in fb_defs
                ),
                columns=["feedback_definition_id", "feedback_json"],
            )

    def insert_feedback(
        self, feedback_result: schema.FeedbackResult
    ) -> schema.FeedbackResultID:
        # TODO: thread safety

        _feedback_result = orm.FeedbackResult.parse(
            feedback_result, redact_keys=self.redact_keys
        )
        with self.Session.begin() as session:
            if session.query(orm.FeedbackResult) \
                    .filter_by(feedback_result_id=feedback_result.feedback_result_id).first():
                session.merge(_feedback_result)  # update existing
            else:
                session.merge(
                    _feedback_result
                )  # insert new result # .add was not thread safe

            status = FeedbackResultStatus(_feedback_result.status)

            if status == FeedbackResultStatus.DONE:
                icon = UNICODE_CHECK
            elif status == FeedbackResultStatus.RUNNING:
                icon = UNICODE_HOURGLASS
            elif status == FeedbackResultStatus.NONE:
                icon = UNICODE_CLOCK
            elif status == FeedbackResultStatus.FAILED:
                icon = UNICODE_STOP
            else:
                icon = "???"

            logger.info(
                f"{icon} feedback result {_feedback_result.name} {status.name} {_feedback_result.feedback_result_id}"
            )

            return _feedback_result.feedback_result_id

    def _feedback_query(
        self,
        count: bool = False,
        shuffle: bool = False,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ):
        if count:
            q = func.count(orm.FeedbackResult.feedback_result_id)
        else:
            q = select(orm.FeedbackResult)

        if record_id:
            q = q.filter_by(record_id=record_id)

        if feedback_result_id:
            q = q.filter_by(feedback_result_id=feedback_result_id)

        if feedback_definition_id:
            q = q.filter_by(feedback_definition_id=feedback_definition_id)

        if status:
            if isinstance(status, FeedbackResultStatus):
                status = [status.value]
            q = q.filter(
                orm.FeedbackResult.status.in_([s.value for s in status])
            )
        if last_ts_before:
            q = q.filter(
                orm.FeedbackResult.last_ts < last_ts_before.timestamp()
            )

        if offset is not None:
            q = q.offset(offset)

        if limit is not None:
            q = q.limit(limit)

        if shuffle:
            q = q.order_by(func.random())

        return q

    def get_feedback_count_by_status(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Dict[FeedbackResultStatus, int]:
        """
        Get the number of feedback results that match the given criteria grouped by status.
        """

        with self.Session.begin() as session:
            q = self._feedback_query(
                count=True, **locals_except("self", "session")
            )

            results = session.query(orm.FeedbackResult.status,
                                    q).group_by(orm.FeedbackResult.status)

            return {FeedbackResultStatus(row[0]): row[1] for row in results}

    def get_feedback(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        See abstract trulens_eval.db:DB.get_feedback for documentation.
        """

        with self.Session.begin() as session:
            q = self._feedback_query(**locals_except("self", "session"))

            results = (row[0] for row in session.execute(q))

            return _extract_feedback_results(results)

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        with self.Session.begin() as session:
            stmt = select(orm.AppDefinition)
            if app_ids:
                stmt = stmt.where(orm.AppDefinition.app_id.in_(app_ids))
            apps = (row[0] for row in session.execute(stmt))
            return AppsExtractor().get_df_and_cols(apps)


# Use this Perf for missing Perfs.
# TODO: Migrate the database instead.
no_perf = schema.Perf(
    start_time=datetime.min, end_time=datetime.min
).model_dump()


def _extract_feedback_results(
    results: Iterable[orm.FeedbackResult]
) -> pd.DataFrame:

    def _extract(_result: orm.FeedbackResult):
        app_json = json.loads(_result.record.app.app_json)
        _type = schema.AppDefinition.model_validate(app_json).root_class

        return (
            _result.record_id,
            _result.feedback_result_id,
            _result.feedback_definition_id,
            _result.last_ts,
            FeedbackResultStatus(_result.status),
            _result.error,
            _result.name,
            _result.result,
            _result.multi_result,
            _result.cost_json,  # why is cost_json not parsed?
            json.loads(_result.record.perf_json)
            if _result.record.perf_json != MIGRATION_UNKNOWN_STR else no_perf,
            json.loads(_result.calls_json)["calls"],
            json.loads(_result.feedback_definition.feedback_json)
            if _result.feedback_definition is not None else None,
            json.loads(_result.record.record_json),
            app_json,
            _type,
        )

    df = pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            'record_id',
            'feedback_result_id',
            'feedback_definition_id',
            'last_ts',
            'status',
            'error',
            'fname',
            'result',
            'multi_result',
            'cost_json',
            'perf_json',
            'calls_json',
            'feedback_json',
            'record_json',
            'app_json',
            "type",
        ],
    )
    df["latency"] = _extract_latency(df["perf_json"])
    df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
    return df


def _extract_latency(
    series: Iterable[Union[str, dict, schema.Perf]]
) -> pd.Series:

    def _extract(perf_json: Union[str, dict, schema.Perf]) -> int:
        if perf_json == MIGRATION_UNKNOWN_STR:
            return np.nan

        if isinstance(perf_json, str):
            perf_json = json.loads(perf_json)

        if isinstance(perf_json, dict):
            perf_json = schema.Perf.model_validate(perf_json)

        if isinstance(perf_json, schema.Perf):
            return perf_json.latency.seconds

        if perf_json is None:
            return 0

        raise ValueError(f"Failed to parse perf_json: {perf_json}")

    return pd.Series(data=(_extract(p) for p in series))


def _extract_tokens_and_cost(cost_json: pd.Series) -> pd.DataFrame:

    def _extract(_cost_json: Union[str, dict]) -> Tuple[int, float]:
        if isinstance(_cost_json, str):
            _cost_json = json.loads(_cost_json)
        if _cost_json is not None:
            cost = schema.Cost(**_cost_json)
        else:
            cost = schema.Cost()
        return cost.n_tokens, cost.cost

    return pd.DataFrame(
        data=(_extract(c) for c in cost_json),
        columns=["total_tokens", "total_cost"],
    )


class AppsExtractor:
    app_cols = ["app_id", "app_json", "type"]
    rec_cols = [
        "record_id", "input", "output", "tags", "record_json", "cost_json",
        "perf_json", "ts"
    ]
    extra_cols = ["latency", "total_tokens", "total_cost"]
    all_cols = app_cols + rec_cols + extra_cols

    def __init__(self):
        self.feedback_columns = set()

    def get_df_and_cols(
        self, apps: Iterable[orm.AppDefinition]
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        df = pd.concat(self.extract_apps(apps))
        df["latency"] = _extract_latency(df["perf_json"])
        df.reset_index(
            drop=True, inplace=True
        )  # prevent index mismatch on the horizontal concat that follows
        df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
        return df, list(self.feedback_columns)

    def extract_apps(
        self, apps: Iterable[orm.AppDefinition]
    ) -> Iterable[pd.DataFrame]:
        yield pd.DataFrame(
            [], columns=self.app_cols + self.rec_cols
        )  # prevent empty iterator
        for _app in apps:
            if _recs := _app.records:
                df = pd.DataFrame(data=self.extract_records(_recs))

                for col in self.app_cols:
                    if col == "type":
                        # Previous DBs did not contain entire app so we cannot
                        # deserialize AppDefinition here unless we fix prior DBs
                        # in migration. Because of this, loading just the
                        # `root_class` here.
                        df[col] = str(
                            Class.model_validate(
                                json.loads(_app.app_json).get('root_class')
                            )
                        )
                    else:
                        df[col] = getattr(_app, col)

                yield df

    def extract_records(self,
                        records: Iterable[orm.Record]) -> Iterable[pd.Series]:
        for _rec in records:
            calls = defaultdict(list)
            values = defaultdict(list)

            for _res in _rec.feedback_results:
                calls[_res.name].append(json.loads(_res.calls_json)["calls"])
                if _res.multi_result is not None and (multi_result :=
                                                      json.loads(
                                                          _res.multi_result
                                                      )) is not None:
                    for key, val in multi_result.items():
                        if val is not None:  # avoid getting Nones into np.mean
                            name = f"{_res.name}:::{key}"
                            values[name] = val
                            self.feedback_columns.add(name)
                elif _res.result is not None:  # avoid getting Nones into np.mean
                    values[_res.name].append(_res.result)
                    self.feedback_columns.add(_res.name)

            row = {
                **{
                    k: np.mean(v) for k, v in values.items()
                },
                **{
                    k + "_calls": flatten(v) for k, v in calls.items()
                },
            }

            for col in self.rec_cols:
                row[col] = datetime.fromtimestamp(
                    _rec.ts
                ).isoformat() if col == "ts" else getattr(_rec, col)

            yield row


def flatten(nested: Iterable[Iterable[Any]]) -> List[Any]:

    def _flatten(_nested):
        for iterable in _nested:
            for element in iterable:
                yield element

    return list(_flatten(nested))
