import logging
import shutil
import sqlite3
import warnings
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Sequence

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from trulens_eval import schema
from trulens_eval.db import DB
from trulens_eval.db import LocalSQLite
from trulens_eval.db_v2 import models
from trulens_eval.db_v2.migrations import migrate_db, DbRevisions
from trulens_eval.db_v2.utils import for_all_methods, run_before

logger = logging.getLogger(__name__)


def is_legacy_sqlite(engine: Engine) -> bool:
    """Check if DB is an existing file-based SQLite
    that was never handled with Alembic"""
    return (
            engine.url.drivername.startswith("sqlite")  # The database type is SQLite
            and Path(engine.url.database).is_file()  # The database location is an existing file
            and DbRevisions.load(engine).current is None  # Alembic could not determine the revision
    )


def is_memory_sqlite(engine: Engine) -> bool:
    """Check if DB is an in-memory SQLite instance"""
    return (
            engine.url.drivername.startswith("sqlite")  # The database type is SQLite
            and engine.url.database == ":memory:"  # The database storage is in memory
    )


def migrate_legacy_sqlite(engine: Engine):
    """Migrate legacy file-based SQLite to the latest Alembic revision:

    Migration plan:
        1. Make sure that original database is at the latest legacy schema
        2. Create empty staging database at the first Alembic revision
        3. Copy records from original database to staging
        4. Migrate staging database to the latest Alembic revision
        5. Replace original database file with the staging one

    Assumptions:
        1. The latest legacy schema is not identical to the first Alembic revision,
           so it is not safe to apply the Alembic migration scripts directly (e.g.:
           TEXT fields used as primary key needed to be changed to VARCHAR due to
           limitations in MySQL).
        2. The latest legacy schema is similar enough to the first Alembic revision,
           and SQLite typing is lenient enough, so that the data exported from the
           original database can be loaded into the staging one.
    """
    # 1. Make sure that original database is at the latest legacy schema
    assert is_legacy_sqlite(engine)
    original_file = Path(engine.url.database)
    logger.info("Handling legacy SQLite file: %s", original_file)
    logger.debug("Applying legacy migration scripts")
    LocalSQLite(filename=original_file).migrate_database()

    with TemporaryDirectory() as tmp:
        # 2. Create empty staging database at first Alembic revision
        stg_file = Path(tmp).joinpath("migration-staging.sqlite")
        logger.debug("Creating staging DB at %s", stg_file)
        stg_engine = create_engine(f"sqlite:///{stg_file}")
        migrate_db(stg_engine, revision="1")

        # 3. Copy records from original database to staging
        src_conn = sqlite3.connect(original_file)
        tgt_conn = sqlite3.connect(stg_file)
        for table in ["apps"]:  # legacy_db.TABLES:  # TODO: copy other tables too
            logger.debug("Copying table '%s'", table)
            df = pd.read_sql_query(f"SELECT * FROM {table}", src_conn)
            df.to_sql(table, tgt_conn, index=False, if_exists="append")

        # 4. Migrate staging database to the latest Alembic revision
        logger.debug("Applying Alembic migration scripts")
        migrate_db(stg_engine, revision="head")

        # 5. Replace original database file with the staging one
        logger.debug("Replacing database file at %s", original_file)
        shutil.copyfile(stg_file, original_file)


def _copy_database(src_url: str, tgt_url: str):
    """Copy all data from a source database to an EMPTY target database.

    Important considerations:
        - All source data will be appended to the target tables,
          so it is important that the target database is empty.

        - Will fail if the databases are not at the latest schema revision.
          That can be fixed with `Tru(database_url="...").migrate_database()`

        - Might fail if the target database enforces relationship constraints,
          because then the order of inserting data matters.

        - This process is NOT transactional, so it is highly recommended
          that the databases are NOT used by anyone while this process runs.
    """
    src = SqlAlchemyDB.from_db_url(src_url)
    check_db_revision(src.engine)

    tgt = SqlAlchemyDB.from_db_url(tgt_url)
    check_db_revision(tgt.engine)

    for table in ["apps"]:  # legacy_db.TABLES:  # TODO: copy other tables too
        with src.engine.begin() as src_conn:
            with tgt.engine.begin() as tgt_conn:
                df = pd.read_sql_query(f"SELECT * FROM {table}", src_conn)
                df.to_sql(table, tgt_conn, index=False, if_exists="append")


def check_db_revision(engine: Engine):
    """Check if database schema is at the expected revision"""
    if is_legacy_sqlite(engine):
        logger.info("Found legacy SQLite file: %s", engine.url.database)
        raise DatabaseVersionException.behind()

    revisions = DbRevisions.load(engine)

    if revisions.current is None:
        logger.debug("Creating database")
        migrate_db(engine, revision="head")  # create automatically if it doesn't exist
    elif revisions.in_sync:
        logger.debug("Database schema is up to date: %s", revisions)
    elif revisions.behind:
        raise DatabaseVersionException.behind()
    elif revisions.ahead:
        raise DatabaseVersionException.ahead()
    else:
        raise NotImplementedError(f"Cannot handle database revisions: {revisions}")


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
            if _app := session.query(models.App).filter_by(app_id=app.app_id).first():
                _app.app_json = app.json()
            else:
                _app = models.App.parse(app)
                session.add(_app)
            return _app.app_id

    def insert_feedback_definition(self, feedback_definition: schema.FeedbackDefinition) -> schema.FeedbackDefinitionID:
        pass  # TODO: impl

    def insert_feedback(self, feedback_result: schema.FeedbackResult) -> schema.FeedbackResultID:
        pass  # TODO: impl

    def get_records_and_feedback(self, app_ids: List[str]) -> Tuple[pd.DataFrame, Sequence[str]]:
        pass  # TODO: impl


class DatabaseVersionException(Exception):
    class Reason(Enum):
        AHEAD = 1
        BEHIND = 2

    def __init__(self, msg: str, reason: Reason):
        self.reason = reason
        super().__init__(msg)

    @classmethod
    def ahead(cls):
        return cls(
            "Database schema is ahead of the expected revision. "
            "Please update to a later release of `trulens_eval`",
            cls.Reason.AHEAD
        )

    @classmethod
    def behind(cls):
        return cls(
            "Database schema is behind the expected revision. "
            "Please upgrade it by running `tru.migrate_database()`",
            cls.Reason.BEHIND
        )
