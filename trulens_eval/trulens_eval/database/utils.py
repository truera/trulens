import inspect
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List, Callable, Union

import pandas as pd
from sqlalchemy import Engine, create_engine

from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.migrations import DbRevisions, upgrade_db
from trulens_eval.db import LocalSQLite

logger = logging.getLogger(__name__)


def for_all_methods(decorator, _except: Optional[List[str]] = None):
    """Applies decorator to all methods except classmethods,
    private methods and the ones specified with `_except`"""
    def decorate(cls):
        for attr_name, attr in cls.__dict__.items():  # does not include classmethods
            if not inspect.isfunction(attr):
                continue  # skips non-method attributes
            if attr_name.startswith("_"):
                continue  # skips private methods
            if _except is not None and attr_name in _except:
                continue
            logger.debug(f"Decorating {attr_name}")
            setattr(cls, attr_name, decorator(attr))
        return cls

    return decorate


def run_before(callback: Callable):
    """Create decorator to run the callback before the function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            callback(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_legacy_sqlite(engine: Engine) -> bool:
    """Check if DB is an existing file-based SQLite
    created with the `LocalSQLite` implementation

    Will also return True for an empty, brand-new SQLite file,
    but applying the legacy migrations on the empty database
    before the Alembic scripts will not hurt, so this should
    be a harmless false positive.

    Checking for the existence of the `meta` table is not safe
    because in trulens_eval 0.1.2 the meta table may not exist.
    """

    if not engine.url.drivername.startswith("sqlite"):
        return False

    return DbRevisions.load(engine).current is None


def is_memory_sqlite(engine: Engine) -> bool:
    """Check if DB is an in-memory SQLite instance"""
    return (
            engine.url.drivername.startswith("sqlite")  # The database type is SQLite
            and engine.url.database == ":memory:"  # The database storage is in memory
    )


def check_db_revision(engine: Engine):
    """Check if database schema is at the expected revision"""
    if is_legacy_sqlite(engine):
        logger.info("Found legacy SQLite file: %s", engine.url.database)
        raise DatabaseVersionException.behind()

    revisions = DbRevisions.load(engine)

    if revisions.current is None:
        logger.debug("Creating database")
        upgrade_db(engine, revision="head")  # create automatically if it doesn't exist
    elif revisions.in_sync:
        logger.debug("Database schema is up to date: %s", revisions)
    elif revisions.behind:
        raise DatabaseVersionException.behind()
    elif revisions.ahead:
        raise DatabaseVersionException.ahead()
    else:
        raise NotImplementedError(f"Cannot handle database revisions: {revisions}")


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
        upgrade_db(stg_engine, revision="1")

        # 3. Copy records from original database to staging
        src_conn = sqlite3.connect(original_file)
        tgt_conn = sqlite3.connect(stg_file)
        for table in ["apps", "feedback_defs", "records", "feedbacks"]:
            logger.info("Copying table '%s'", table)
            df = pd.read_sql(f"SELECT * FROM {table}", src_conn)
            for col in ["ts", "last_ts"]:
                if col in df:
                    df[col] = df[col].apply(lambda ts: coerce_ts(ts).timestamp())
            logger.debug("\n\n%s\n", df.head())
            df.to_sql(table, tgt_conn, index=False, if_exists="append")

        # 4. Migrate staging database to the latest Alembic revision
        logger.debug("Applying Alembic migration scripts")
        upgrade_db(stg_engine, revision="head")

        # 5. Replace original database file with the staging one
        logger.debug("Replacing database file at %s", original_file)
        shutil.copyfile(stg_file, original_file)


def coerce_ts(ts: Union[datetime, str, int, float]) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    raise ValueError(f"Cannot coerce to datetime: {ts}")


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
    from trulens_eval.database.sqlalchemy_db import SqlAlchemyDB

    src = SqlAlchemyDB.from_db_url(src_url)
    check_db_revision(src.engine)

    tgt = SqlAlchemyDB.from_db_url(tgt_url)
    check_db_revision(tgt.engine)

    for table in ["apps", "feedback_defs", "records", "feedbacks"]:
        with src.engine.begin() as src_conn:
            with tgt.engine.begin() as tgt_conn:
                df = pd.read_sql(f"SELECT * FROM {table}", src_conn)
                df.to_sql(table, tgt_conn, index=False, if_exists="append")
