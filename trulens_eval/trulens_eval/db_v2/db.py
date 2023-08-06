import logging
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Sequence

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from trulens_eval.db import DB
from trulens_eval.db import LocalSQLite
from trulens_eval.db_v2.migrations import migrate_db, set_db_revision, DbRevisions
from trulens_eval.db_v2.utils import for_all_methods, run_before
from trulens_eval.schema import FeedbackResult, FeedbackResultID, FeedbackDefinition, FeedbackDefinitionID, \
    AppDefinition, AppID, Record, RecordID

logger = logging.getLogger(__name__)


def is_legacy_sqlite(engine: Engine):
    """Check if DB is an existing file-based SQLite
    that was never handled with Alembic"""
    return (
            engine.url.drivername.startswith("sqlite")  # The database type is SQLite
            and Path(engine.url.database).is_file()  # The database location is an existing file
            and DbRevisions.load(engine).current is None  # Alembic could not determine the revision
    )


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

    def migrate_database(self):
        """Migrate database schema to the latest revision

        If database is a legacy SQLite file, this will:
            1. Apply the legacy migration scripts in order to
               bring the database to the latest legacy schema.
            2. Stamp the database with revision "1", so Alembic
               will recognize that it is at the first revision.
            3. Apply any remaining Alembic migration scripts
               from current revision up to "head" (latest).

        In any other scenarios, only step 3 is applied.
        If the database is new, step 3 will bring it from
        revision `None` to "1", then move on until "head".
        """
        if is_legacy_sqlite(self.engine):
            logger.info("Handling legacy SQLite file: %s", self.engine.url.database)
            LocalSQLite(filename=Path(self.engine.url.database)).migrate_database()  # step 1
            set_db_revision(self.engine, revision="1")  # step 2
        migrate_db(self.engine, revision="head")  # step 3

    @classmethod
    def from_db_url(cls, url: str, **kwargs):
        return cls(engine=create_engine(url, **kwargs))

    def reset_database(self):
        raise NotImplementedError(
            f"Resetting the database is not implemented for `{self.__class__}`. "
            "Please perform this operation by connecting to the database directly"
        )

    def insert_record(self, record: Record) -> RecordID:
        pass  # TODO: impl

    def insert_app(self, app: AppDefinition) -> AppID:
        pass  # TODO: impl

    def insert_feedback_definition(self, feedback_definition: FeedbackDefinition) -> FeedbackDefinitionID:
        pass  # TODO: impl

    def insert_feedback(self, feedback_result: FeedbackResult) -> FeedbackResultID:
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
