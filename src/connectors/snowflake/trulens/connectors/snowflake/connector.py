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
from trulens.core.database import base as core_db
from trulens.core.database.base import DB
from trulens.core.database.connector.base import DBConnector
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.schema.types import AppID
from trulens.core.schema import app as app_schema
from trulens.core.utils import python as python_utils
from trulens.core import __version__ as trulens_version

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
            connection_parameters["schema"] = schema
        else:
            snowpark_connection_parameters = {
                "account": snowpark_session.get_current_account(),
                "user": snowpark_session.get_current_user(),
                "database": snowpark_session.get_current_database(),
                "schema": snowpark_session.get_current_schema(),
                "warehouse": snowpark_session.get_current_warehouse(),
                "role": snowpark_session.get_current_role(),
            }
            missing_snowpark_params = []
            mismatched_kwargs = []
            for k, v in snowpark_connection_parameters.items():
                if not v:
                    missing_snowpark_params.append(k)
                if connection_parameters[k] is None:
                    connection_parameters[k] = v
                elif connection_parameters[k] != v:
                    mismatched_kwargs.append(k)

            if missing_snowpark_params:
                raise ValueError(
                    f"Connection parameters missing from provided `snowpark_session`: {missing_snowpark_params}"
                )
            if mismatched_kwargs:
                raise ValueError(
                    f"Connection parameters mismatch between provided `snowpark_session` and args passed to `SnowflakeConnector`: {mismatched_kwargs}"
                )

            if connection_parameters["password"] is None:
                # NOTE: user passwords are inaccessible from the `snowpark_session` object.
                logger.warning(
                    "Running the TruLens dashboard requires providing a `password` to the `SnowflakeConnector`."
                )
                connection_parameters["password"] = "password"

        self._init_with_snowpark_session(
            snowpark_session,
            init_server_side,
            database_redact_keys,
            database_prefix,
            database_args,
            database_check_revision,
            connection_parameters,
        )

    def _init_with_snowpark_session(
        self,
        snowpark_session: Session,
        init_server_side: bool,
        database_redact_keys: bool,
        database_prefix: Optional[str],
        database_args: Optional[Dict[str, Any]],
        database_check_revision: bool,
        connection_parameters: Dict[str, Optional[str]],
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

        database_url = URL(
            **connection_parameters,
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
            database_prefix or core_db.DEFAULT_DATABASE_PREFIX
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
        
        # Add "trulens_workspace_version" tag to the current schema
        schema = snowpark_session.get_current_schema()
        snowpark_session.sql(f"ALTER SCHEMA {schema} SET TAG trulens_workspace_version = {trulens_version}")

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
