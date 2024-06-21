from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import logging
from sqlite3 import OperationalError
from typing import (
    Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
)
import warnings

from alembic.ddl.impl import DefaultImpl
import numpy as np
import pandas as pd
from pydantic import Field
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sql_text

from trulens_eval import app as mod_app
from trulens_eval.database import base as mod_db
from trulens_eval.database import orm as mod_orm
from trulens_eval.database.base import DB
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.migrations.data import data_migrate
from trulens_eval.database.utils import \
    check_db_revision as alembic_check_db_revision
from trulens_eval.database.utils import is_legacy_sqlite
from trulens_eval.database.utils import is_memory_sqlite
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import text
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSONized
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK
from trulens_eval.utils.text import UNICODE_HOURGLASS
from trulens_eval.utils.text import UNICODE_STOP

logger = logging.getLogger(__name__)


class SnowflakeImpl(DefaultImpl):
    __dialect__ = 'snowflake'


class SQLAlchemyDB(DB):
    """Database implemented using sqlalchemy.
    
    See abstract class [DB][trulens_eval.database.base.DB] for method reference.
    """

    table_prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
    """The prefix to use for all table names. 
    
    [DB][trulens_eval.database.base.DB] interface requirement.
    """

    engine_params: dict = Field(default_factory=dict)
    """Sqlalchemy-related engine params."""

    session_params: dict = Field(default_factory=dict)
    """Sqlalchemy-related session."""

    engine: Optional[Engine] = None
    """Sqlalchemy engine."""

    session: Optional[sessionmaker] = None
    """Sqlalchemy session(maker)."""

    model_config: ClassVar[dict] = {'arbitrary_types_allowed': True}

    orm: Type[mod_orm.ORM]
    """
    Container of all the ORM classes for this database.

    This should be set to a subclass of
    [ORM][trulens_eval.database.orm.ORM] upon initialization.
    """

    def __init__(
        self,
        redact_keys: bool = mod_db.DEFAULT_DATABASE_REDACT_KEYS,
        table_prefix: str = mod_db.DEFAULT_DATABASE_PREFIX,
        **kwargs: Dict[str, Any]
    ):
        super().__init__(
            redact_keys=redact_keys,
            table_prefix=table_prefix,
            orm=mod_orm.make_orm_for_prefix(table_prefix=table_prefix),
            **kwargs
        )
        self._reload_engine()
        if is_memory_sqlite(self.engine):
            warnings.warn(
                UserWarning(
                    "SQLite in-memory may not be threadsafe. "
                    "See https://www.sqlite.org/threadsafe.html"
                )
            )

    def _reload_engine(self):
        self.engine = create_engine(**self.engine_params)
        self.session = sessionmaker(self.engine, **self.session_params)

    @classmethod
    def from_tru_args(
        cls,
        database_url: Optional[str] = None,
        database_file: Optional[str] = None,
        database_redact_keys: Optional[bool] = mod_db.
        DEFAULT_DATABASE_REDACT_KEYS,
        database_prefix: Optional[str] = mod_db.DEFAULT_DATABASE_PREFIX,
        **kwargs: Dict[str, Any]
    ) -> SQLAlchemyDB:
        """Process database-related configuration provided to the [Tru][trulens_eval.tru.Tru] class to
        create a database.
        
        Emits warnings if appropriate.
        """

        if None not in (database_url, database_file):
            raise ValueError(
                "Please specify at most one of `database_url` and `database_file`"
            )

        if database_file:
            warnings.warn(
                (
                    "`database_file` is deprecated, "
                    "use `database_url` instead as in `database_url='sqlite:///filename'."
                ),
                DeprecationWarning,
                stacklevel=2
            )

        if database_url is None:
            database_url = f"sqlite:///{database_file or mod_db.DEFAULT_DATABASE_FILE}"

        if 'table_prefix' not in kwargs:
            kwargs['table_prefix'] = database_prefix

        if 'redact_keys' not in kwargs:
            kwargs['redact_keys'] = database_redact_keys

        new_db: DB = SQLAlchemyDB.from_db_url(database_url, **kwargs)

        print(
            "%s Tru initialized with db url %s ." %
            (text.UNICODE_SQUID, new_db.engine.url)
        )
        if database_redact_keys:
            print(
                f"{text.UNICODE_LOCK} Secret keys will not be included in the database."
            )
        else:
            print(
                f"{text.UNICODE_STOP} Secret keys may be written to the database. "
                "See the `database_redact_keys` option of `Tru` to prevent this."
            )

        return new_db

    @classmethod
    def from_db_url(cls, url: str, **kwargs: Dict[str, Any]) -> SQLAlchemyDB:
        """
        Create a database for the given url.

        Args:
            url: The database url. This includes database type.

            kwargs: Additional arguments to pass to the database constructor.

        Returns:
            A database instance.
        """

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

        return cls(engine_params=engine_params, **kwargs)

    def check_db_revision(self):
        """See
        [DB.check_db_revision][trulens_eval.database.base.DB.check_db_revision]."""

        if self.engine is None:
            raise ValueError("Database engine not initialized.")

        alembic_check_db_revision(self.engine, self.table_prefix)

    def migrate_database(self, prior_prefix: Optional[str] = None):
        """See [DB.migrate_database][trulens_eval.database.base.DB.migrate_database]."""

        if self.engine is None:
            raise ValueError("Database engine not initialized.")

        try:
            # Expect to get the the behind exception.
            alembic_check_db_revision(
                self.engine,
                prefix=self.table_prefix,
                prior_prefix=prior_prefix
            )

            # If we get here, our db revision does not need upgrade.
            logger.warning("Database does not need migration.")

        except DatabaseVersionException as e:
            if e.reason == DatabaseVersionException.Reason.BEHIND:

                revisions = DbRevisions.load(self.engine)
                from_version = revisions.current
                ### SCHEMA MIGRATION ###
                if is_legacy_sqlite(self.engine):
                    raise RuntimeError(
                        "Migrating legacy sqlite database is no longer supported. "
                        "A database reset is required. This will delete all existing data: "
                        "`tru.reset_database()`."
                    ) from e

                else:
                    ## TODO Create backups here. This is not sqlalchemy's strong suit: https://stackoverflow.com/questions/56990946/how-to-backup-up-a-sqlalchmey-database
                    ### We might allow migrate_database to take a backup url (and suggest user to supply if not supplied ala `tru.migrate_database(backup_db_url="...")`)
                    ### We might try copy_database as a backup, but it would need to automatically handle clearing the db, and also current implementation requires migrate to run first.
                    ### A valid backup would need to be able to copy an old version, not the newest version
                    upgrade_db(
                        self.engine, revision="head", prefix=self.table_prefix
                    )

                self._reload_engine(
                )  # let sqlalchemy recognize the migrated schema

                ### DATA MIGRATION ###
                data_migrate(self, from_version)
                return

            elif e.reason == DatabaseVersionException.Reason.AHEAD:
                # Rethrow the ahead message suggesting to upgrade trulens_eval.
                raise e

            elif e.reason == DatabaseVersionException.Reason.RECONFIGURED:
                # Rename table to change prefix.

                prior_prefix = e.prior_prefix

                logger.warning(
                    "Renaming tables from prefix \"%s\" to \"%s\".",
                    prior_prefix, self.table_prefix
                )
                # logger.warning("Please ignore these warnings: \"SAWarning: This declarative base already contains...\"")

                with self.engine.connect() as c:
                    for table_name in ['alembic_version'
                                      ] + [c._table_base_name
                                           for c in self.orm.registry.values()
                                           if hasattr(c, "_table_base_name")]:
                        old_version_table = f"{prior_prefix}{table_name}"
                        new_version_table = f"{self.table_prefix}{table_name}"

                        logger.warning(
                            "  %s -> %s", old_version_table, new_version_table
                        )

                        c.execute(
                            sql_text(
                                """ALTER TABLE %s RENAME TO %s;""" %
                                (old_version_table, new_version_table)
                            )
                        )

            else:
                # TODO: better message here for unhandled cases?
                raise e

    def reset_database(self):
        """See [DB.reset_database][trulens_eval.database.base.DB.reset_database]."""

        #meta = MetaData()
        meta = self.orm.metadata  #
        meta.reflect(bind=self.engine)
        meta.drop_all(bind=self.engine)

        self.migrate_database()

    def insert_record(
        self, record: mod_record_schema.Record
    ) -> mod_types_schema.RecordID:
        """See [DB.insert_record][trulens_eval.database.base.DB.insert_record]."""
        # TODO: thread safety

        _rec = self.orm.Record.parse(record, redact_keys=self.redact_keys)
        with self.session.begin() as session:
            if session.query(self.orm.Record
                            ).filter_by(record_id=record.record_id).first():
                session.merge(_rec)  # update existing
            else:
                session.merge(_rec)  # add new record # .add was not thread safe

            logger.info("{UNICODE_CHECK} added record %s", _rec.record_id)

            return _rec.record_id

    def get_app(
        self, app_id: mod_types_schema.AppID
    ) -> Optional[JSONized[mod_app.App]]:
        """See [DB.get_app][trulens_eval.database.base.DB.get_app]."""

        with self.session.begin() as session:
            if _app := session.query(self.orm.AppDefinition
                                    ).filter_by(app_id=app_id).first():
                return json.loads(_app.app_json)

    def get_apps(self) -> Iterable[JSON]:
        """See [DB.get_apps][trulens_eval.database.base.DB.get_apps]."""

        with self.session.begin() as session:
            for _app in session.query(self.orm.AppDefinition):
                yield json.loads(_app.app_json)

    def insert_app(
        self, app: mod_app_schema.AppDefinition
    ) -> mod_types_schema.AppID:
        """See [DB.insert_app][trulens_eval.database.base.DB.insert_app]."""

        # TODO: thread safety

        with self.session.begin() as session:
            if _app := session.query(self.orm.AppDefinition
                                    ).filter_by(app_id=app.app_id).first():

                _app.app_json = app.model_dump_json()
            else:
                _app = self.orm.AppDefinition.parse(
                    app, redact_keys=self.redact_keys
                )
                session.merge(_app)  # .add was not thread safe

            logger.info("%s added app %s", UNICODE_CHECK, _app.app_id)

            return _app.app_id

    def delete_app(self, app_id: mod_types_schema.AppID) -> None:
        """
        Deletes an app from the database based on its app_id.

        Args:
            app_id (schema.AppID): The unique identifier of the app to be deleted.
        """
        with self.Session.begin() as session:
            _app = session.query(orm.AppDefinition).filter_by(app_id=app_id
                                                             ).first()
            if _app:
                session.delete(_app)
                logger.info(f"{UNICODE_CHECK} deleted app {app_id}")
            else:
                logger.warning(f"App {app_id} not found for deletion.")

    def insert_feedback_definition(
        self, feedback_definition: mod_feedback_schema.FeedbackDefinition
    ) -> mod_types_schema.FeedbackDefinitionID:
        """See [DB.insert_feedback_definition][trulens_eval.database.base.DB.insert_feedback_definition]."""

        # TODO: thread safety

        with self.session.begin() as session:
            if _fb_def := session.query(self.orm.FeedbackDefinition) \
                    .filter_by(feedback_definition_id=feedback_definition.feedback_definition_id) \
                    .first():
                _fb_def.app_json = feedback_definition.model_dump_json()
            else:
                _fb_def = self.orm.FeedbackDefinition.parse(
                    feedback_definition, redact_keys=self.redact_keys
                )
                session.merge(_fb_def)  # .add was not thread safe

            logger.info(
                "%s added feedback definition %s", UNICODE_CHECK,
                _fb_def.feedback_definition_id
            )

            return _fb_def.feedback_definition_id

    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                        ] = None
    ) -> pd.DataFrame:
        """See [DB.get_feedback_defs][trulens_eval.database.base.DB.get_feedback_defs]."""

        with self.session.begin() as session:
            q = select(self.orm.FeedbackDefinition)
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
        self, feedback_result: mod_feedback_schema.FeedbackResult
    ) -> mod_types_schema.FeedbackResultID:
        """See [DB.insert_feedback][trulens_eval.database.base.DB.insert_feedback]."""

        # TODO: thread safety

        _feedback_result = self.orm.FeedbackResult.parse(
            feedback_result, redact_keys=self.redact_keys
        )
        with self.session.begin() as session:
            if session.query(self.orm.FeedbackResult) \
                    .filter_by(feedback_result_id=feedback_result.feedback_result_id).first():
                session.merge(_feedback_result)  # update existing
            else:
                session.merge(
                    _feedback_result
                )  # insert new result # .add was not thread safe

            status = mod_feedback_schema.FeedbackResultStatus(
                _feedback_result.status
            )

            if status == mod_feedback_schema.FeedbackResultStatus.DONE:
                icon = UNICODE_CHECK
            elif status == mod_feedback_schema.FeedbackResultStatus.RUNNING:
                icon = UNICODE_HOURGLASS
            elif status == mod_feedback_schema.FeedbackResultStatus.NONE:
                icon = UNICODE_CLOCK
            elif status == mod_feedback_schema.FeedbackResultStatus.FAILED:
                icon = UNICODE_STOP
            else:
                icon = "???"

            logger.info(
                "%s feedback result %s %s %s", icon, _feedback_result.name,
                status.name, _feedback_result.feedback_result_id
            )

            return _feedback_result.feedback_result_id

    def _feedback_query(
        self,
        count: bool = False,
        shuffle: bool = False,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                        ] = None,
        status: Optional[
            Union[mod_feedback_schema.FeedbackResultStatus,
                  Sequence[mod_feedback_schema.FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ):
        if count:
            q = func.count(self.orm.FeedbackResult.feedback_result_id)
        else:
            q = select(self.orm.FeedbackResult)

        if record_id:
            q = q.filter_by(record_id=record_id)

        if feedback_result_id:
            q = q.filter_by(feedback_result_id=feedback_result_id)

        if feedback_definition_id:
            q = q.filter_by(feedback_definition_id=feedback_definition_id)

        if status:
            if isinstance(status, mod_feedback_schema.FeedbackResultStatus):
                status = [status.value]
            q = q.filter(
                self.orm.FeedbackResult.status.in_([s.value for s in status])
            )
        if last_ts_before:
            q = q.filter(
                self.orm.FeedbackResult.last_ts < last_ts_before.timestamp()
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
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                        ] = None,
        status: Optional[
            Union[mod_feedback_schema.FeedbackResultStatus,
                  Sequence[mod_feedback_schema.FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Dict[mod_feedback_schema.FeedbackResultStatus, int]:
        """See [DB.get_feedback_count_by_status][trulens_eval.database.base.DB.get_feedback_count_by_status]."""

        with self.session.begin() as session:
            q = self._feedback_query(
                count=True, **locals_except("self", "session")
            )

            results = session.query(self.orm.FeedbackResult.status,
                                    q).group_by(self.orm.FeedbackResult.status)

            return {
                mod_feedback_schema.FeedbackResultStatus(row[0]): row[1]
                for row in results
            }

    def get_feedback(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                        ] = None,
        status: Optional[
            Union[mod_feedback_schema.FeedbackResultStatus,
                  Sequence[mod_feedback_schema.FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = False
    ) -> pd.DataFrame:
        """See [DB.get_feedback][trulens_eval.database.base.DB.get_feedback]."""

        with self.session.begin() as session:
            q = self._feedback_query(**locals_except("self", "session"))

            results = (row[0] for row in session.execute(q))

            return _extract_feedback_results(results)

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """See [DB.get_records_and_feedback][trulens_eval.database.base.DB.get_records_and_feedback]."""

        # TODO: Add pagination to this method. Currently the joinedload in
        # select below disables lazy loading of records which will be a problem
        # for large databases without the use of pagination.

        with self.session.begin() as session:
            stmt = select(self.orm.AppDefinition).options(
                joinedload(self.orm.AppDefinition.records)\
                .joinedload(self.orm.Record.feedback_results)
            )

            if app_ids:
                stmt = stmt.where(self.orm.AppDefinition.app_id.in_(app_ids))

            ex = session.execute(stmt).unique()  # unique needed for joinedload
            apps = (row[0] for row in ex)

            return AppsExtractor().get_df_and_cols(apps)


# Use this Perf for missing Perfs.
# TODO: Migrate the database instead.
no_perf = mod_base_schema.Perf.min().model_dump()


def _extract_feedback_results(
    results: Iterable[orm.FeedbackResult]
) -> pd.DataFrame:

    def _extract(_result: self.orm.FeedbackResult):
        app_json = json.loads(_result.record.app.app_json)
        _type = mod_app_schema.AppDefinition.model_validate(app_json).root_class

        return (
            _result.record_id,
            _result.feedback_result_id,
            _result.feedback_definition_id,
            _result.last_ts,
            mod_feedback_schema.FeedbackResultStatus(_result.status),
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
    series: Iterable[Union[str, dict, mod_base_schema.Perf]]
) -> pd.Series:

    def _extract(perf_json: Union[str, dict, mod_base_schema.Perf]) -> int:
        if perf_json == MIGRATION_UNKNOWN_STR:
            return np.nan

        if isinstance(perf_json, str):
            perf_json = json.loads(perf_json)

        if isinstance(perf_json, dict):
            perf_json = mod_base_schema.Perf.model_validate(perf_json)

        if isinstance(perf_json, mod_base_schema.Perf):
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
            cost = mod_base_schema.Cost(**_cost_json)
        else:
            cost = mod_base_schema.Cost()
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
            try:
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
            except OperationalError as e:
                print(
                    "Error encountered while attempting to retrieve an app. "
                    "This issue may stem from a corrupted database."
                )
                print(f"Error details: {e}")

    def extract_records(self,
                        records: Iterable[orm.Record]) -> Iterable[pd.Series]:

        for _rec in records:
            calls = defaultdict(list)
            values = defaultdict(list)

            try:
                for _res in _rec.feedback_results:

                    calls[_res.name].append(
                        json.loads(_res.calls_json)["calls"]
                    )
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
                    **{k: np.mean(v) for k, v in values.items()},
                    **{k + "_calls": flatten(v) for k, v in calls.items()},
                }

                for col in self.rec_cols:
                    row[col] = datetime.fromtimestamp(
                        _rec.ts
                    ).isoformat() if col == "ts" else getattr(_rec, col)

                yield row

            except Exception as e:
                # Handling unexpected errors, possibly due to database issues.
                print(
                    "Error encountered while attempting to retrieve feedback results. "
                    "This issue may stem from a corrupted database."
                )
                print(f"Error details: {e}")


def flatten(nested: Iterable[Iterable[Any]]) -> List[Any]:

    def _flatten(_nested):
        for iterable in _nested:
            for element in iterable:
                yield element

    return list(_flatten(nested))
