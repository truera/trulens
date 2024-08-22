from __future__ import annotations

from functools import cached_property
import logging
import re
from typing import (
    Any,
    Dict,
    Optional,
    Union,
)

from trulens.core.database.base import DB
from trulens.core.database.connector.base import DBConnector
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.utils.python import OpaqueWrapper

from snowflake.core import CreateMode
from snowflake.core import Root
from snowflake.core.schema import Schema
from snowflake.snowpark import Session
from snowflake.sqlalchemy import URL

logger = logging.getLogger(__name__)


class SnowflakeConnector(DBConnector):
    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        database_name: str,
        schema_name: str,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
    ):
        database_args = database_args or {}

        self._validate_schema_name(schema_name)
        database_url = self._create_snowflake_database_url(
            account=account,
            user=user,
            password=password,
            database=database_name,
            schema=schema_name,
            warehouse=warehouse,
            role=role,
        )
        database_args.update({
            k: v
            for k, v in {
                "database_url": database_url,
                "database_redact_keys": database_redact_keys,
                "database_prefix": database_prefix,
            }.items()
            if v is not None
        })
        self._db: Union[SQLAlchemyDB, OpaqueWrapper] = (
            SQLAlchemyDB.from_tru_args(**database_args)
        )

        if database_check_revision:
            try:
                self._db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self._db = OpaqueWrapper(obj=self._db, e=e)

    @classmethod
    def _validate_schema_name(cls, name: str) -> None:
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                "`name` must contain only alphanumeric and underscore characters!"
            )

    @classmethod
    def _create_snowflake_database_url(cls, **kwargs) -> str:
        cls._create_snowflake_schema_if_not_exists(**kwargs)
        return URL(**kwargs)

    @classmethod
    def _create_snowflake_schema_if_not_exists(cls, **kwargs):
        session = Session.builder.configs(**kwargs).create()
        root = Root(session)
        schema_name = kwargs.get("schema", None)
        if schema_name is None:
            raise ValueError("Schema name must be provided.")

        database_name = kwargs.get("database", None)
        if database_name is None:
            raise ValueError("Database name must be provided.")
        schema = Schema(name=schema_name)
        root.databases[database_name].schemas.create(
            schema, mode=CreateMode.if_not_exists
        )

    @cached_property
    def db(self) -> DB:
        if isinstance(self._db, OpaqueWrapper):
            self._db = self._db.unwrap()
        if not isinstance(self._db, DB):
            raise RuntimeError("Unhandled database type.")
        return self._db
