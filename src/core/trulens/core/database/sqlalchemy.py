from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import logging
from sqlite3 import OperationalError
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import warnings

from alembic.ddl.impl import DefaultImpl
import numpy as np
import pandas as pd
from pydantic import Field
import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sql_text
from trulens.core.database import base as mod_db
from trulens.core.database import orm as mod_orm
from trulens.core.database.base import DB
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens.core.database.migrations import DbRevisions
from trulens.core.database.migrations import upgrade_db
from trulens.core.database.migrations.data import data_migrate
from trulens.core.database.utils import (
    check_db_revision as alembic_check_db_revision,
)
from trulens.core.database.utils import is_legacy_sqlite
from trulens.core.database.utils import is_memory_sqlite
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import base as mod_base_schema
from trulens.core.schema import dataset as mod_dataset_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import groundtruth as mod_groundtruth_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import text
from trulens.core.utils.pyschema import Class
from trulens.core.utils.python import locals_except
from trulens.core.utils.serial import JSON
from trulens.core.utils.serial import JSONized
from trulens.core.utils.text import UNICODE_CHECK
from trulens.core.utils.text import UNICODE_CLOCK
from trulens.core.utils.text import UNICODE_HOURGLASS
from trulens.core.utils.text import UNICODE_STOP

logger = logging.getLogger(__name__)


class SnowflakeImpl(DefaultImpl):
    __dialect__ = "snowflake"


class SQLAlchemyDB(DB):
    """Database implemented using sqlalchemy.

    See abstract class [DB][trulens.core.database.base.DB] for method reference.
    """

    table_prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
    """The prefix to use for all table names.

    [DB][trulens.core.database.base.DB] interface requirement.
    """

    engine_params: dict = Field(default_factory=dict)
    """Sqlalchemy-related engine params."""

    session_params: dict = Field(default_factory=dict)
    """Sqlalchemy-related session."""

    engine: Optional[sa.Engine] = None
    """Sqlalchemy engine."""

    session: Optional[sessionmaker] = None
    """Sqlalchemy session(maker)."""

    model_config: ClassVar[dict] = {"arbitrary_types_allowed": True}

    orm: Type[mod_orm.ORM]
    """
    Container of all the ORM classes for this database.

    This should be set to a subclass of
    [ORM][trulens.core.database.orm.ORM] upon initialization.
    """

    def __init__(
        self,
        redact_keys: bool = mod_db.DEFAULT_DATABASE_REDACT_KEYS,
        table_prefix: str = mod_db.DEFAULT_DATABASE_PREFIX,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(
            redact_keys=redact_keys,
            table_prefix=table_prefix,
            orm=mod_orm.make_orm_for_prefix(table_prefix=table_prefix),
            **kwargs,
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
        if self.engine is None:
            self.engine = sa.create_engine(**self.engine_params)
        self.session = sessionmaker(self.engine, **self.session_params)

    @classmethod
    def from_tru_args(
        cls,
        database_url: Optional[str] = None,
        database_engine: Optional[sa.Engine] = None,
        database_file: Optional[str] = None,
        database_redact_keys: Optional[
            bool
        ] = mod_db.DEFAULT_DATABASE_REDACT_KEYS,
        database_prefix: Optional[str] = mod_db.DEFAULT_DATABASE_PREFIX,
        **kwargs: Dict[str, Any],
    ) -> SQLAlchemyDB:
        """Process database-related configuration provided to the [Tru][trulens.core.session.TruSession] class to
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
                stacklevel=2,
            )

        if database_url is None:
            database_url = (
                f"sqlite:///{database_file or mod_db.DEFAULT_DATABASE_FILE}"
            )

        if "table_prefix" not in kwargs:
            kwargs["table_prefix"] = database_prefix

        if "redact_keys" not in kwargs:
            kwargs["redact_keys"] = database_redact_keys

        if database_engine is not None:
            new_db: DB = SQLAlchemyDB.from_db_engine(database_engine, **kwargs)
        else:
            new_db: DB = SQLAlchemyDB.from_db_url(database_url, **kwargs)

        print(
            "%s TruSession initialized with db url %s ."
            % (text.UNICODE_SQUID, new_db.engine.url)
        )
        if database_redact_keys:
            print(
                f"{text.UNICODE_LOCK} Secret keys will not be included in the database."
            )
        else:
            print(
                f"{text.UNICODE_STOP} Secret keys may be written to the database. "
                "See the `database_redact_keys` option of `TruSession` to prevent this."
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

        default_engine_params = {
            "url": url,
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
        }

        if not is_memory_sqlite(url=url):
            # These params cannot be given to memory-based sqlite engine.
            default_engine_params["max_overflow"] = 2
            default_engine_params["pool_use_lifo"] = True

        if "engine_params" in kwargs:
            for k, v in default_engine_params.items():
                if k not in kwargs["engine_params"]:
                    kwargs["engine_params"][k] = v
        else:
            kwargs["engine_params"] = default_engine_params

        return cls(**kwargs)

    @classmethod
    def from_db_engine(
        cls, engine: sa.Engine, **kwargs: Dict[str, Any]
    ) -> SQLAlchemyDB:
        """
        Create a database for the given engine.
        Args:
            engine: The database engine.
            kwargs: Additional arguments to pass to the database constructor.
        Returns:
            A database instance.
        """

        return cls(engine=engine, **kwargs)

    def check_db_revision(self):
        """See
        [DB.check_db_revision][trulens.core.database.base.DB.check_db_revision]."""

        if self.engine is None:
            raise ValueError("Database engine not initialized.")

        alembic_check_db_revision(self.engine, self.table_prefix)

    def migrate_database(self, prior_prefix: Optional[str] = None):
        """See [DB.migrate_database][trulens.core.database.base.DB.migrate_database]."""

        if self.engine is None:
            raise ValueError("Database engine not initialized.")

        try:
            # Expect to get the the behind exception.
            alembic_check_db_revision(
                self.engine, prefix=self.table_prefix, prior_prefix=prior_prefix
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
                        "`TruSession.reset_database()`."
                    ) from e

                else:
                    ## TODO Create backups here. This is not sqlalchemy's strong suit: https://stackoverflow.com/questions/56990946/how-to-backup-up-a-sqlalchmey-database
                    ### We might allow migrate_database to take a backup url (and suggest user to supply if not supplied ala `TruSession.migrate_database(backup_db_url="...")`)
                    ### We might try copy_database as a backup, but it would need to automatically handle clearing the db, and also current implementation requires migrate to run first.
                    ### A valid backup would need to be able to copy an old version, not the newest version
                    upgrade_db(
                        self.engine, revision="head", prefix=self.table_prefix
                    )

                self._reload_engine()  # let sqlalchemy recognize the migrated schema

                ### DATA MIGRATION ###
                data_migrate(self, from_version)
                return

            elif e.reason == DatabaseVersionException.Reason.AHEAD:
                # Rethrow the ahead message suggesting to upgrade trulens.
                raise e

            elif e.reason == DatabaseVersionException.Reason.RECONFIGURED:
                # Rename table to change prefix.

                prior_prefix = e.prior_prefix

                logger.warning(
                    'Renaming tables from prefix "%s" to "%s".',
                    prior_prefix,
                    self.table_prefix,
                )
                # logger.warning("Please ignore these warnings: \"SAWarning: This declarative base already contains...\"")

                with self.engine.connect() as c:
                    for table_name in ["alembic_version"] + [
                        c._table_base_name
                        for c in self.orm.registry.values()
                        if hasattr(c, "_table_base_name")
                    ]:
                        old_version_table = f"{prior_prefix}{table_name}"
                        new_version_table = f"{self.table_prefix}{table_name}"

                        logger.warning(
                            "  %s -> %s", old_version_table, new_version_table
                        )

                        c.execute(
                            sql_text(
                                """ALTER TABLE %s RENAME TO %s;"""
                                % (old_version_table, new_version_table)
                            )
                        )

            else:
                # TODO: better message here for unhandled cases?
                raise e

    def reset_database(self):
        """See [DB.reset_database][trulens.core.database.base.DB.reset_database]."""

        # meta = MetaData()
        meta = self.orm.metadata  #
        meta.reflect(bind=self.engine)
        meta.drop_all(bind=self.engine)

        self.migrate_database()

    def insert_record(
        self, record: mod_record_schema.Record
    ) -> mod_types_schema.RecordID:
        """See [DB.insert_record][trulens.core.database.base.DB.insert_record]."""
        # TODO: thread safety

        _rec = self.orm.Record.parse(record, redact_keys=self.redact_keys)
        with self.session.begin() as session:
            if (
                session.query(self.orm.Record)
                .filter_by(record_id=record.record_id)
                .first()
            ):
                session.merge(_rec)  # update existing
            else:
                session.merge(_rec)  # add new record # .add was not thread safe

            logger.info("{UNICODE_CHECK} added record %s", _rec.record_id)

            return _rec.record_id

    def batch_insert_record(
        self, records: List[mod_record_schema.Record]
    ) -> List[mod_types_schema.RecordID]:
        """See [DB.batch_insert_record][trulens_eval.database.base.DB.batch_insert_record]."""
        with self.session.begin() as session:
            records_list = [
                self.orm.Record.parse(r, redact_keys=self.redact_keys)
                for r in records
            ]
            session.bulk_save_objects(records_list)
            logger.info(f"{UNICODE_CHECK} added record batch")
            # return record ids from orm objects
            return [r.record_id for r in records_list]

    def get_app(self, app_id: mod_types_schema.AppID) -> Optional[JSONized]:
        """See [DB.get_app][trulens.core.database.base.DB.get_app]."""

        with self.session.begin() as session:
            if (
                _app := session.query(self.orm.AppDefinition)
                .filter_by(app_id=app_id)
                .first()
            ):
                return json.loads(_app.app_json)

    def get_apps(self) -> Iterable[JSON]:
        """See [DB.get_apps][trulens.core.database.base.DB.get_apps]."""

        with self.session.begin() as session:
            for _app in session.query(self.orm.AppDefinition):
                yield json.loads(_app.app_json)

    def insert_app(
        self, app: mod_app_schema.AppDefinition
    ) -> mod_types_schema.AppID:
        """See [DB.insert_app][trulens.core.database.base.DB.insert_app]."""

        # TODO: thread safety

        with self.session.begin() as session:
            if (
                _app := session.query(self.orm.AppDefinition)
                .filter_by(app_id=app.app_id)
                .first()
            ):
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
        with self.session.begin() as session:
            _app = (
                session.query(self.orm.AppDefinition)
                .filter_by(app_id=app_id)
                .first()
            )
            if _app:
                session.delete(_app)
                logger.info(f"{UNICODE_CHECK} deleted app {app_id}")
            else:
                logger.warning(f"App {app_id} not found for deletion.")

    def insert_feedback_definition(
        self, feedback_definition: mod_feedback_schema.FeedbackDefinition
    ) -> mod_types_schema.FeedbackDefinitionID:
        """See [DB.insert_feedback_definition][trulens.core.database.base.DB.insert_feedback_definition]."""

        # TODO: thread safety

        with self.session.begin() as session:
            if (
                _fb_def := session.query(self.orm.FeedbackDefinition)
                .filter_by(
                    feedback_definition_id=feedback_definition.feedback_definition_id
                )
                .first()
            ):
                _fb_def.app_json = feedback_definition.model_dump_json()
            else:
                _fb_def = self.orm.FeedbackDefinition.parse(
                    feedback_definition, redact_keys=self.redact_keys
                )
                session.merge(_fb_def)  # .add was not thread safe

            logger.info(
                "%s added feedback definition %s",
                UNICODE_CHECK,
                _fb_def.feedback_definition_id,
            )

            return _fb_def.feedback_definition_id

    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
    ) -> pd.DataFrame:
        """See [DB.get_feedback_defs][trulens.core.database.base.DB.get_feedback_defs]."""

        with self.session.begin() as session:
            q = sa.select(self.orm.FeedbackDefinition)
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
        """See [DB.insert_feedback][trulens.core.database.base.DB.insert_feedback]."""

        # TODO: thread safety

        _feedback_result = self.orm.FeedbackResult.parse(
            feedback_result, redact_keys=self.redact_keys
        )
        with self.session.begin() as session:
            if (
                session.query(self.orm.FeedbackResult)
                .filter_by(
                    feedback_result_id=feedback_result.feedback_result_id
                )
                .first()
            ):
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
                "%s feedback result %s %s %s",
                icon,
                _feedback_result.name,
                status.name,
                _feedback_result.feedback_result_id,
            )

            return _feedback_result.feedback_result_id

    def batch_insert_feedback(
        self, feedback_results: List[mod_feedback_schema.FeedbackResult]
    ) -> List[mod_types_schema.FeedbackResultID]:
        """See [DB.batch_insert_feedback][trulens_eval.database.base.DB.batch_insert_feedback]."""
        with self.session.begin() as session:
            feedback_results_list = [
                self.orm.FeedbackResult.parse(f, redact_keys=self.redact_keys)
                for f in feedback_results
            ]
            session.bulk_save_objects(feedback_results_list)
            return [f.feedback_result_id for f in feedback_results_list]

    def _feedback_query(
        self,
        count_by_status: bool = False,
        shuffle: bool = False,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ):
        if count_by_status:
            q = sa.select(
                self.orm.FeedbackResult.status,
                sa.func.count(self.orm.FeedbackResult.feedback_result_id),
            ).group_by(self.orm.FeedbackResult.status)
        else:
            q = sa.select(self.orm.FeedbackResult)

        if record_id:
            q = q.filter_by(record_id=record_id)

        if feedback_result_id:
            q = q.filter_by(feedback_result_id=feedback_result_id)

        if feedback_definition_id:
            q = q.filter_by(feedback_definition_id=feedback_definition_id)

        if (
            run_location is None
            or run_location == mod_feedback_schema.FeedbackRunLocation.IN_APP
        ):
            # For legacy reasons, we handle the IN_APP and NULL/None case as the same.
            q = q.filter(
                sa.or_(
                    self.orm.FeedbackDefinition.run_location.is_(None),
                    self.orm.FeedbackDefinition.run_location
                    == mod_feedback_schema.FeedbackRunLocation.IN_APP.value,
                )
            )
        else:
            q = q.filter(
                self.orm.FeedbackDefinition.run_location == run_location.value
            )
        q = q.filter(
            self.orm.FeedbackResult.feedback_definition_id
            == self.orm.FeedbackDefinition.feedback_definition_id
        )

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
            q = q.order_by(sa.func.random())

        return q

    def get_feedback_count_by_status(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ) -> Dict[mod_feedback_schema.FeedbackResultStatus, int]:
        """See [DB.get_feedback_count_by_status][trulens.core.database.base.DB.get_feedback_count_by_status]."""

        with self.session.begin() as session:
            q = self._feedback_query(
                count_by_status=True,
                **locals_except("self", "session"),
            )
            results = session.execute(q)

            return {
                mod_feedback_schema.FeedbackResultStatus(row[0]): row[1]
                for row in results
            }

    def get_feedback(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ) -> pd.DataFrame:
        """See [DB.get_feedback][trulens.core.database.base.DB.get_feedback]."""

        with self.session.begin() as session:
            q = self._feedback_query(**locals_except("self", "session"))

            results = (row[0] for row in session.execute(q))

            return _extract_feedback_results(results)

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """See [DB.get_records_and_feedback][trulens.core.database.base.DB.get_records_and_feedback]."""

        # TODO: Add pagination to this method. Currently the joinedload in
        # select below disables lazy loading of records which will be a problem
        # for large databases without the use of pagination.

        with self.session.begin() as session:
            stmt = sa.select(self.orm.Record)
            # NOTE: We are selecting records here because offset and limit need
            # to be with respect to those rows instead of AppDefinition or
            # FeedbackResult rows.

            if app_ids:
                stmt = stmt.where(self.orm.Record.app_id.in_(app_ids))

            stmt = stmt.options(joinedload(self.orm.Record.feedback_results))
            # NOTE(piotrm): The joinedload here makes it so that the
            # feedback_results get loaded eagerly instead if lazily when
            # accessed later.

            # TODO(piotrm): The subsequent logic in helper methods end up
            # reading all of the records and feedback_results in order to create
            # a DataFrame so there is no reason to not eagerly get all of this
            # data. Ideally, though, we would be making some sort of lazy
            # DataFrame and then could use the lazy join feature of sqlalchemy.

            stmt = stmt.order_by(self.orm.Record.ts, self.orm.Record.record_id)
            # NOTE: feedback_results order is governed by the order_by on the
            # orm.FeedbackResult.record backref definition. Here, we need to
            # order Records as we did not use an auto join to retrieve them. If
            # records were to be retrieved from AppDefinition.records via auto
            # join, though, the orm backref ordering would be able to take hold.

            stmt = stmt.limit(limit).offset(offset)

            ex = session.execute(stmt).unique()
            # unique needed for joinedload above.

            records = [rec[0] for rec in ex]
            # TODO: Make the iteration of records lazy in some way. See
            # TODO(piotrm) above.

            return AppsExtractor().get_df_and_cols(records=records)

    def insert_ground_truth(
        self, ground_truth: mod_groundtruth_schema.GroundTruth
    ) -> mod_types_schema.GroundTruthID:
        """See [DB.insert_ground_truth][trulens.core.database.base.DB.insert_ground_truth]."""

        # TODO: thread safety
        with self.session.begin() as session:
            if (
                _ground_truth := session.query(self.orm.GroundTruth)
                .filter_by(ground_truth_id=ground_truth.ground_truth_id)
                .first()
            ):
                # Update the existing record for idempotency
                _ground_truth.ground_truth_json = ground_truth.model_dump_json()
            else:
                _ground_truth = self.orm.GroundTruth.parse(
                    ground_truth, redact_keys=self.redact_keys
                )

                session.merge(_ground_truth)

            logger.info(
                f"{UNICODE_CHECK} added ground truth {_ground_truth.ground_truth_id}"
            )

            return _ground_truth.ground_truth_id

    def batch_insert_ground_truth(
        self, ground_truths: List[mod_groundtruth_schema.GroundTruth]
    ) -> List[mod_types_schema.GroundTruthID]:
        """See [DB.batch_insert_ground_truth][trulens_eval.database.base.DB.batch_insert_ground_truth]."""
        with self.session.begin() as session:
            ground_truth_ids = [gt.ground_truth_id for gt in ground_truths]

            # Fetch existing GroundTruth records that match these ids in one query
            existing_ground_truths = (
                session.query(self.orm.GroundTruth)
                .filter(
                    self.orm.GroundTruth.ground_truth_id.in_(ground_truth_ids)
                )
                .all()
            )

            existing_ground_truth_dict = {
                gt.ground_truth_id: gt for gt in existing_ground_truths
            }

            ground_truths_to_insert = []
            for ground_truth in ground_truths:
                if ground_truth.ground_truth_id in existing_ground_truth_dict:
                    existing_record = existing_ground_truth_dict[
                        ground_truth.ground_truth_id
                    ]
                    # Update the existing record for idempotency
                    existing_record.ground_truth_json = (
                        ground_truth.model_dump_json()
                    )
                else:
                    new_ground_truth = self.orm.GroundTruth.parse(
                        ground_truth, redact_keys=self.redact_keys
                    )
                    ground_truths_to_insert.append(new_ground_truth)

            session.bulk_save_objects(ground_truths_to_insert)
            return [gt.ground_truth_id for gt in ground_truths]

    def get_ground_truth(
        self, ground_truth_id: str | None = None
    ) -> Optional[JSONized]:
        """See [DB.get_ground_truth][trulens.core.database.base.DB.get_ground_truth]."""

        with self.session.begin() as session:
            if (
                _ground_truth := session.query(self.orm.GroundTruth)
                .filter_by(ground_truth_id=ground_truth_id)
                .first()
            ):
                return json.loads(_ground_truth)

    def get_ground_truths_by_dataset(
        self, dataset_name: str
    ) -> pd.DataFrame | None:
        """See [DB.get_ground_truths_by_dataset][trulens.core.database.base.DB.get_ground_truths_by_dataset]."""
        with self.session.begin() as session:
            q = sa.select(self.orm.Dataset)
            all_datasets = (row[0] for row in session.execute(q))
            df = None
            for dataset in all_datasets:
                dataset_json = json.loads(dataset.dataset_json)
                if (
                    "name" in dataset_json
                    and dataset_json["name"] == dataset_name
                ):
                    q = sa.select(self.orm.GroundTruth).filter_by(
                        dataset_id=dataset.dataset_id
                    )
                    results = (row[0] for row in session.execute(q))

                    if df is None:
                        df = _extract_ground_truths(results)
                    else:
                        df = pd.concat([df, _extract_ground_truths(results)])
            return df
            # TODO: use a generator instead of a list? (for large datasets)

    def insert_dataset(
        self, dataset: mod_dataset_schema.Dataset
    ) -> mod_types_schema.DatasetID:
        """See [DB.insert_dataset][trulens.core.database.base.DB.insert_dataset]."""

        with self.session.begin() as session:
            if (
                _dataset := session.query(self.orm.Dataset)
                .filter_by(dataset_id=dataset.dataset_id)
                .first()
            ):
                # Update the existing record for idempotency
                _dataset.dataset_json = dataset.model_dump_json()
            else:
                _dataset = self.orm.Dataset.parse(
                    dataset, redact_keys=self.redact_keys
                )
                session.merge(_dataset)

            logger.info(f"{UNICODE_CHECK} added dataset {_dataset.dataset_id}")

            return _dataset.dataset_id

    def get_datasets(self) -> pd.DataFrame:
        """See [DB.get_datasets][trulens.core.database.base.DB.get_datasets]."""

        with self.session.begin() as session:
            results = session.query(self.orm.Dataset)

            return pd.DataFrame(
                data=((ds.dataset_id, ds.name, ds.meta) for ds in results),
                columns=["dataset_id", "name", "meta"],
            )


# Use this Perf for missing Perfs.
# TODO: Migrate the database instead.
no_perf = mod_base_schema.Perf.min().model_dump()


def _extract_feedback_results(
    results: Iterable["mod_orm.FeedbackResult"],
) -> pd.DataFrame:
    def _extract(_result: "mod_orm.FeedbackResult"):
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
            if _result.record.perf_json != MIGRATION_UNKNOWN_STR
            else no_perf,
            json.loads(_result.calls_json)["calls"],
            json.loads(_result.feedback_definition.feedback_json)
            if _result.feedback_definition is not None
            else None,
            json.loads(_result.record.record_json),
            app_json,
            _type,
        )

    df = pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            "record_id",
            "feedback_result_id",
            "feedback_definition_id",
            "last_ts",
            "status",
            "error",
            "fname",
            "result",
            "multi_result",
            "cost_json",
            "perf_json",
            "calls_json",
            "feedback_json",
            "record_json",
            "app_json",
            "type",
        ],
    )
    df["latency"] = _extract_latency(df["perf_json"])
    df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
    return df


def _extract_latency(
    series: Iterable[Union[str, dict, mod_base_schema.Perf]],
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
    def _extract(_cost_json: Union[str, dict]) -> Tuple[int, float, str]:
        if isinstance(_cost_json, str):
            _cost_json = json.loads(_cost_json)
        if _cost_json is not None:
            cost = mod_base_schema.Cost(**_cost_json)
        else:
            cost = mod_base_schema.Cost()
        return cost.n_tokens, cost.cost, cost.cost_currency

    return pd.DataFrame(
        data=(_extract(c) for c in cost_json),
        columns=["total_tokens", "total_cost", "cost_currency"],
    )


def _extract_ground_truths(
    results: Iterable["mod_orm.GroundTruth"],
) -> pd.DataFrame:
    def _extract(_result: "mod_orm.GroundTruth"):
        ground_truth_json = json.loads(_result.ground_truth_json)

        return (
            _result.ground_truth_id,
            _result.dataset_id,
            ground_truth_json["query"],
            ground_truth_json["query_id"],
            ground_truth_json["expected_response"],
            ground_truth_json["expected_chunks"],
            ground_truth_json["meta"],
        )

    return pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            "ground_truth_id",
            "dataset_id",
            "query",
            "query_id",
            "expected_response",
            "expected_chunks",
            "meta",
        ],
    )


class AppsExtractor:
    """Utilities for creating dataframes from orm instances."""

    app_cols = ["app_id", "app_json", "type"]
    rec_cols = [
        "record_id",
        "input",
        "output",
        "tags",
        "record_json",
        "cost_json",
        "perf_json",
        "ts",
    ]
    extra_cols = ["latency", "total_tokens", "total_cost"]
    all_cols = app_cols + rec_cols + extra_cols

    def __init__(self):
        self.feedback_columns = set()

    def get_df_and_cols(
        self,
        apps: Optional[List["mod_orm.ORM.AppDefinition"]] = None,
        records: Optional[List["mod_orm.ORM.Record"]] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Produces a records dataframe which joins in information from apps and
        feedback results.

        Args:
            apps: If given, includes all records of all of the apps in this
                iterable.

            records: If given, includes only these records. Mutually exclusive
                with `apps`.
        """

        assert (
            apps is None or records is None
        ), "`apps` and `records` are mutually exclusive"

        if apps is not None:
            df = pd.concat(self.extract_apps(apps))

        elif records is not None:
            apps = {record.app for record in records}
            df = pd.concat(self.extract_apps(apps=apps, records=records))

        else:
            raise ValueError("'apps` or `records` must be provided")

        df["latency"] = _extract_latency(df["perf_json"])
        df.reset_index(
            drop=True, inplace=True
        )  # prevent index mismatch on the horizontal concat that follows
        df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
        return df, list(self.feedback_columns)

    def extract_apps(
        self,
        apps: Iterable["mod_orm.ORM.AppDefinition"],
        records: Optional[List["mod_orm.ORM.Record"]] = None,
    ) -> Iterable[pd.DataFrame]:
        """
        Creates record rows with app information.

        TODO: The means for enumerating records in this method is not ideal as
        it does a lot of filtering.
        """

        yield pd.DataFrame(
            [], columns=self.app_cols + self.rec_cols
        )  # prevent empty iterator
        for _app in apps:
            try:
                if records is None:
                    # If records not provided, get all of them for `_app`.
                    _recs = _app.records
                else:
                    # Otherwise get only the ones in `records`. WARNING: Avoid
                    # using _app.records here as doing so might get all of the
                    # records even the ones not in `records`
                    _recs = (
                        record
                        for record in records
                        if record.app_id == _app.app_id
                    )

                if _recs:
                    df = pd.DataFrame(data=self.extract_records(_recs))

                    for col in self.app_cols:
                        if col == "type":
                            # Previous DBs did not contain entire app so we cannot
                            # deserialize AppDefinition here unless we fix prior DBs
                            # in migration. Because of this, loading just the
                            # `root_class` here.

                            df[col] = str(
                                Class.model_validate(
                                    json.loads(_app.app_json).get("root_class")
                                )
                            )

                        else:
                            df[col] = getattr(_app, col)

                    df["app_name"] = _app.app_name
                    df["app_version"] = _app.app_version
                    yield df
            except OperationalError as e:
                print(
                    "Error encountered while attempting to retrieve an app. "
                    "This issue may stem from a corrupted database."
                )
                print(f"Error details: {e}")

    def extract_records(
        self, records: Iterable["mod_orm.ORM.Record"]
    ) -> Iterable[pd.Series]:
        for _rec in records:
            calls = defaultdict(list)
            values = defaultdict(list)
            feedback_cost = {}
            try:
                for _res in _rec.feedback_results:
                    calls[_res.name].append(
                        json.loads(_res.calls_json)["calls"]
                    )

                    feedback_usage = json.loads(_res.cost_json)
                    cost_currency = feedback_usage.get("cost_currency", "USD")

                    if "cost" in feedback_usage:
                        feedback_cost[
                            f"{_res.name} feedback cost in {cost_currency}"
                        ] = feedback_usage["cost"]

                    if (
                        _res.multi_result is not None
                        and (multi_result := json.loads(_res.multi_result))
                        is not None
                    ):
                        for key, val in multi_result.items():
                            if (
                                val is not None
                            ):  # avoid getting Nones into np.mean
                                name = f"{_res.name}:::{key}"
                                values[name] = val
                                self.feedback_columns.add(name)
                    elif (
                        _res.result is not None
                    ):  # avoid getting Nones into np.mean
                        values[_res.name].append(_res.result)
                        self.feedback_columns.add(_res.name)

                row = {
                    **{k: np.mean(v) for k, v in values.items()},
                    **{k + "_calls": flatten(v) for k, v in calls.items()},
                    **{k: v for k, v in feedback_cost.items()},
                }

                for col in self.rec_cols:
                    row[col] = (
                        datetime.fromtimestamp(_rec.ts).isoformat()
                        if col == "ts"
                        else getattr(_rec, col)
                    )

                yield row

            except Exception as e:
                # Handling unexpected errors, possibly due to database issues.
                print(
                    "Error encountered while attempting to retrieve feedback results. "
                    "This issue may stem from a corrupted database."
                )
                print(f"Error details: {e}")


def flatten(nested: Iterable[Iterable[Any]]) -> List[Any]:
    def _flatten(
        _nested: Iterable[Iterable[Any]],
    ) -> Generator[Any, None, None]:
        for iterable in _nested:
            yield from iterable

    return list(_flatten(nested))
