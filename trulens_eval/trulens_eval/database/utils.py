from datetime import datetime
import logging
from pprint import pformat
from typing import Optional, Union

import pandas as pd
import sqlalchemy
from sqlalchemy import Engine
from sqlalchemy import inspect as sql_inspect

from trulens_eval.database import base as mod_db
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import upgrade_db

logger = logging.getLogger(__name__)


def is_legacy_sqlite(engine: Engine) -> bool:
    """Check if DB is an existing file-based SQLite created with the legacy
    `LocalSQLite` implementation.
    
    This database was removed since trulens_eval 0.29.0 .
    """

    inspector = sql_inspect(engine)
    tables = list(inspector.get_table_names())

    if len(tables) == 0:
        # brand new db, not even initialized yet
        return False

    version_tables = [t for t in tables if t.endswith("alembic_version")]

    return len(version_tables) == 0


def is_memory_sqlite(
    engine: Optional[Engine] = None,
    url: Optional[Union[sqlalchemy.engine.URL, str]] = None
) -> bool:
    """Check if DB is an in-memory SQLite instance.

    Either engine or url can be provided.
    """

    if isinstance(engine, Engine):
        url = engine.url

    elif isinstance(url, sqlalchemy.engine.URL):
        pass

    elif isinstance(url, str):
        url = sqlalchemy.engine.make_url(url)

    else:
        raise ValueError("Either engine or url must be provided")

    return (
        # The database type is SQLite
        url.drivername.startswith("sqlite")

        # The database storage is in memory
        and url.database == ":memory:"
    )


def check_db_revision(
    engine: Engine,
    prefix: str = mod_db.DEFAULT_DATABASE_PREFIX,
    prior_prefix: Optional[str] = None
):
    """
    Check if database schema is at the expected revision.

    Args:
        engine: SQLAlchemy engine to check.

        prefix: Prefix used for table names including alembic_version in the
            current code.

        prior_prefix: Table prefix used in the previous version of the
            database. Before this configuration was an option, the prefix was
            equivalent to "".
    """

    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")

    if prefix == prior_prefix:
        raise ValueError(
            "prior_prefix and prefix canot be the same. Use None for prior_prefix if it is unknown."
        )

    ins = sqlalchemy.inspect(engine)
    tables = ins.get_table_names()

    # Get all tables we could have made for alembic version. Other apps might
    # also have made these though.
    version_tables = [t for t in tables if t.endswith("alembic_version")]

    if prior_prefix is not None:
        # Check if tables using the old/empty prefix exist.
        if prior_prefix + "alembic_version" in version_tables:
            raise DatabaseVersionException.reconfigured(
                prior_prefix=prior_prefix
            )
    else:
        # Check if the new/expected version table exists.

        if prefix + "alembic_version" not in version_tables:
            # If not, lets try to figure out the prior prefix.

            if len(version_tables) > 0:

                if len(version_tables) > 1:
                    # Cannot figure out prior prefix if there is more than one
                    # version table.
                    raise ValueError(
                        f"Found multiple alembic_version tables: {version_tables}. "
                        "Cannot determine prior prefix. "
                        "Please specify it using the `prior_prefix` argument."
                    )

                # Guess prior prefix as the single one with version table name.
                raise DatabaseVersionException.reconfigured(
                    prior_prefix=version_tables[0].
                    replace("alembic_version", "")
                )

    if is_legacy_sqlite(engine):
        logger.info("Found legacy SQLite file: %s", engine.url)
        raise DatabaseVersionException.behind()

    revisions = DbRevisions.load(engine, prefix=prefix)

    if revisions.current is None:
        logger.debug("Creating database")
        upgrade_db(
            engine, revision="head", prefix=prefix
        )  # create automatically if it doesn't exist

    elif revisions.in_sync:
        logger.debug("Database schema is up to date: %s", revisions)

    elif revisions.behind:
        raise DatabaseVersionException.behind()

    elif revisions.ahead:
        raise DatabaseVersionException.ahead()

    else:
        raise NotImplementedError(
            f"Cannot handle database revisions: {revisions}"
        )


def coerce_ts(ts: Union[datetime, str, int, float]) -> datetime:
    """Coerce various forms of timestamp into datetime."""

    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)

    raise ValueError(f"Cannot coerce to datetime: {ts}")


def copy_database(
    src_url: str,
    tgt_url: str,
    src_prefix: str,  # = mod_db.DEFAULT_DATABASE_PREFIX,
    tgt_prefix: str,  # = mod_db.DEFAULT_DATABASE_PREFIX
):
    """Copy all data from a source database to an EMPTY target database.

    Important considerations:
    
    - All source data will be appended to the target tables, so it is
        important that the target database is empty.

    - Will fail if the databases are not at the latest schema revision. That
        can be fixed with `Tru(database_url="...", database_prefix="...").migrate_database()`

    - Might fail if the target database enforces relationship constraints,
        because then the order of inserting data matters.

    - This process is NOT transactional, so it is highly recommended that
        the databases are NOT used by anyone while this process runs.
    """

    # Avoids circular imports.
    from trulens_eval.database.sqlalchemy import SQLAlchemyDB

    src = SQLAlchemyDB.from_db_url(src_url, table_prefix=src_prefix)
    check_db_revision(src.engine, prefix=src_prefix)

    tgt = SQLAlchemyDB.from_db_url(tgt_url, table_prefix=tgt_prefix)
    check_db_revision(tgt.engine, prefix=tgt_prefix)

    print("Source database:")
    print(pformat(src))

    print("Target database:")
    print(pformat(tgt))

    for k, source_table_class in src.orm.registry.items():
        # ["apps", "feedback_defs", "records", "feedbacks"]:

        if not hasattr(source_table_class, "_table_base_name"):
            continue

        target_table_class = tgt.orm.registry.get(k)

        with src.engine.begin() as src_conn:

            with tgt.engine.begin() as tgt_conn:

                df = pd.read_sql(
                    f"SELECT * FROM {source_table_class.__tablename__}",
                    src_conn
                )
                df.to_sql(
                    target_table_class.__tablename__,
                    tgt_conn,
                    index=False,
                    if_exists="append"
                )

                print(
                    f"Copied {len(df)} rows from {source_table_class.__tablename__} in source {target_table_class.__tablename__} in target."
                )
