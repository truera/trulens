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

from trulens.connectors.snowflake.utils.server_side_evaluation_artifacts import (
    ServerSideEvaluationArtifacts,
)
from trulens.core.database import base as mod_db
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
        database: str,
        schema: str,
        warehouse: str,
        role: str,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
    ):
        database_args = database_args or {}

        self._validate_schema_name(schema)
        database_url = self._create_snowflake_database_url(
            account=account,
            user=user,
            password=password,
            database=database,
            schema=schema,
            warehouse=warehouse,
            role=role,
        )
        database_args.update({
            k: v
            for k, v in {
                "database_url": database_url,
                "database_redact_keys": database_redact_keys,
            }.items()
            if v is not None
        })
        database_args["database_prefix"] = (
            database_prefix or mod_db.DEFAULT_DATABASE_PREFIX
        )
        self._db: Union[SQLAlchemyDB, OpaqueWrapper] = (
            SQLAlchemyDB.from_tru_args(**database_args)
        )

        if database_check_revision:
            try:
                self._db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self._db = OpaqueWrapper(obj=self._db, e=e)

        self._initialize_snowflake_server_side_feedback_evaluations(
            account,
            user,
            password,
            database,
            schema,
            warehouse,
            role,
            database_args["database_prefix"],
        )

    def _initialize_snowflake_server_side_feedback_evaluations(
        self,
        account: str,
        user: str,
        password: str,
        database: str,
        schema: str,
        warehouse: str,
        role: str,
        database_prefix: str,
    ):
        connection_parameters = {
            "account": account,
            "user": user,
            "password": password,
            "database": database,
            "schema": schema,
            "warehouse": warehouse,
            "role": role,
        }
        with Session.builder.configs(connection_parameters).create() as session:
            ServerSideEvaluationArtifacts(
                session,
                account,
                user,
                database,
                schema,
                warehouse,
                role,
                database_prefix,
            ).set_up_all()

    @classmethod
    def _validate_schema_name(cls, name: str) -> None:
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                "`name` must contain only alphanumeric and underscore characters!"
            )

    @classmethod
    def _create_snowflake_database_url(cls, **kwargs) -> str:
        kwargs = {k: v for k, v in kwargs.items() if v}
        cls._create_snowflake_schema_if_not_exists(kwargs)
        return URL(**kwargs)

    @classmethod
    def _create_snowflake_schema_if_not_exists(
        cls, connection_parameters: Dict[str, str]
    ):
        with Session.builder.configs(connection_parameters).create() as session:
            root = Root(session)
            schema_name = connection_parameters.get("schema", None)
            if schema_name is None:
                raise ValueError("Schema name must be provided.")

            database_name = connection_parameters.get("database", None)
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
