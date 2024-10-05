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
from trulens.core.utils import python as python_utils

from snowflake.core import CreateMode
from snowflake.core import Root
from snowflake.core.schema import Schema
from snowflake.snowpark import Session
from snowflake.sqlalchemy import URL

logger = logging.getLogger(__name__)


class SnowflakeConnector(DBConnector):
    """Connector to snowflake databases."""

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        snowpark_session: Optional[Session] = None,
        init_server_side: bool = False,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
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
        if snowpark_session is None:
            kwargs_to_set = []
            for k, v in connection_parameters.items():
                if v is None:
                    kwargs_to_set.append(k)
            if kwargs_to_set:
                raise ValueError(
                    f"If not supplying `snowpark_session` then must set `{kwargs_to_set}`!"
                )
            del connection_parameters["schema"]
            snowpark_session = Session.builder.configs(
                connection_parameters
            ).create()
            self._validate_schema_name(schema)
            self._create_snowflake_schema_if_not_exists(
                snowpark_session, database, schema
            )
            snowpark_session.use_schema(schema)
            self._init_with_snowpark_session(
                snowpark_session,
                init_server_side,
                database_redact_keys,
                database_prefix,
                database_args,
                database_check_revision,
            )
        else:
            kwargs_to_not_set = []
            for k, v in connection_parameters.items():
                if v is not None:
                    kwargs_to_not_set.append(k)
            if kwargs_to_not_set:
                raise ValueError(
                    f"Cannot supply both `snowpark_session` and `{kwargs_to_not_set}`!"
                )
            self._init_with_snowpark_session(
                snowpark_session,
                init_server_side,
                database_redact_keys,
                database_prefix,
                database_args,
                database_check_revision,
            )

    def _init_with_snowpark_session(
        self,
        snowpark_session: Session,
        init_server_side: bool,
        database_redact_keys: bool,
        database_prefix: Optional[str],
        database_args: Optional[Dict[str, Any]],
        database_check_revision: bool,
    ):
        database_args = database_args or {}
        if "engine_params" not in database_args:
            database_args["engine_params"] = {}
        if "creator" in database_args["engine_params"]:
            raise ValueError(
                "Cannot set `database_args['engine_params']['creator']!"
            )
        database_args["engine_params"]["creator"] = (
            lambda: snowpark_session.connection
        )
        if "paramstyle" in database_args["engine_params"]:
            raise ValueError(
                "Cannot set `database_args['engine_params']['paramstyle']!"
            )
        database_args["engine_params"]["paramstyle"] = "qmark"

        required_settings = {
            "account": snowpark_session.get_current_account(),
            "user": snowpark_session.get_current_user(),
            "database": snowpark_session.get_current_database(),
            "schema": snowpark_session.get_current_schema(),
            "warehouse": snowpark_session.get_current_warehouse(),
            "role": snowpark_session.get_current_role(),
        }
        for k, v in required_settings.items():
            if not v:
                raise ValueError(f"`{k}` not set in `snowpark_session`!")

        database_url = URL(
            account=snowpark_session.get_current_account(),
            user=snowpark_session.get_current_user(),
            password="password",
            database=snowpark_session.get_current_database(),
            schema=snowpark_session.get_current_schema(),
            warehouse=snowpark_session.get_current_warehouse(),
            role=snowpark_session.get_current_role(),
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
        self._db: Union[SQLAlchemyDB, python_utils.OpaqueWrapper] = (
            SQLAlchemyDB.from_tru_args(**database_args)
        )

        if database_check_revision:
            try:
                self._db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self._db = python_utils.OpaqueWrapper(obj=self._db, e=e)

        if init_server_side:
            ServerSideEvaluationArtifacts(
                snowpark_session, database_args["database_prefix"]
            ).set_up_all()

    @classmethod
    def _validate_schema_name(cls, name: str) -> None:
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                "`name` must contain only alphanumeric and underscore characters!"
            )

    @classmethod
    def _create_snowflake_schema_if_not_exists(
        cls,
        snowpark_session: Session,
        database_name: str,
        schema_name: str,
    ):
        root = Root(snowpark_session)
        schema = Schema(name=schema_name)
        root.databases[database_name].schemas.create(
            schema, mode=CreateMode.if_not_exists
        )

    @cached_property
    def db(self) -> DB:
        if isinstance(self._db, python_utils.OpaqueWrapper):
            self._db = self._db.unwrap()
        if not isinstance(self._db, DB):
            raise RuntimeError("Unhandled database type.")
        return self._db
