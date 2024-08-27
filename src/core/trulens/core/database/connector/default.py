from __future__ import annotations

from functools import cached_property
import logging
from typing import (
    Any,
    Dict,
    Optional,
    Union,
)

import sqlalchemy as sa
from trulens.core.database.base import DB
from trulens.core.database.connector.base import DBConnector
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.utils.python import OpaqueWrapper

logger = logging.getLogger(__name__)


class DefaultDBConnector(DBConnector):
    def __init__(
        self,
        database: Optional[DB] = None,
        database_url: Optional[str] = None,
        database_engine: Optional[sa.Engine] = None,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
    ):
        """Create a default DB connector backed by a database. To connect to an existing database, one of `database`, `database_url`, or `database_engine` must be provided.

        Args:
            database: (Optional[trulens.core.database.base.DB]): The database object to use.
            database_url (Optional[str]): The database URL to connect to. To connect to a local file-based SQLite database, use `sqlite:///path/to/database.db`.
            database_engine (Optional[sa.Engine]): The SQLAlchemy engine object to use.
            database_redact_keys (bool): Whether to redact keys in the database.
            database_prefix (Optional[str]): The database prefix to use to separate tables in the database.
            database_args (Optional[Dict[str, Any]]): Additional arguments to pass to the database.
            database_check_revision (bool): Whether to compare the database revision with the expected TruLens revision.

        """
        self._db: Union[DB, OpaqueWrapper]
        database_args = database_args or {}

        if isinstance(database, DB):
            self._db = database
        elif database is None:
            database_args.update({
                k: v
                for k, v in {
                    "database_url": database_url,
                    "database_engine": database_engine,
                    "database_redact_keys": database_redact_keys,
                    "database_prefix": database_prefix,
                }.items()
                if v is not None
            })
            self._db = SQLAlchemyDB.from_tru_args(**database_args)
        else:
            raise ValueError(
                "`database` must be a `trulens.core.database.base.DB` instance."
            )

        if database_check_revision:
            try:
                self._db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self._db = OpaqueWrapper(obj=self._db, e=e)

    @cached_property
    def db(self) -> DB:
        if isinstance(self._db, OpaqueWrapper):
            self._db = self._db.unwrap()
        if not isinstance(self._db, DB):
            raise RuntimeError("Unhandled database type.")
        return self._db
