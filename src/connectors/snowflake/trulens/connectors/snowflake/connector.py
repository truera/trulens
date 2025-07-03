from __future__ import annotations

from functools import cached_property
import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from trulens.connectors.snowflake.dao.enums import ObjectType
from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
from trulens.connectors.snowflake.dao.run import RunDao
from trulens.connectors.snowflake.utils.server_side_evaluation_artifacts import (
    ServerSideEvaluationArtifacts,
)
from trulens.connectors.snowflake.utils.sis_dashboard_artifacts import (
    SiSDashboardArtifacts,
)
from trulens.core.database import base as core_db
from trulens.core.database.base import DB
from trulens.core.database.connector.base import DBConnector
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.utils import python as python_utils

from snowflake.snowpark import Session
from snowflake.sqlalchemy import URL

logger = logging.getLogger(__name__)

# [HACK!] To have sqlalchemy.JSON work with Snowflake, we need to monkey patch
# the SnowflakeDialect to have the JSON serializer and deserializer set to None.
# This is because by default, SQLAlchemy will set these correctly if they're set
# to None.
try:
    from snowflake.sqlalchemy.snowdialect import SnowflakeDialect

    for attr in ["_json_deserializer", "_json_serializer"]:
        if not hasattr(SnowflakeDialect, attr):
            setattr(SnowflakeDialect, attr, None)
except ImportError:
    pass


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
        protocol: Optional[str] = "https",
        port: Optional[int] = 443,
        host: Optional[str] = None,
        snowpark_session: Optional[Session] = None,
        init_server_side: bool = False,
        init_server_side_with_staged_packages: bool = False,
        init_sis_dashboard: bool = False,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
        use_account_event_table: bool = True,
    ):
        connection_parameters = {
            "account": account,
            "user": user,
            "password": password,
            "database": database,
            "schema": schema,
            "warehouse": warehouse,
            "role": role,
            "protocol": protocol,
            "port": port,
        }

        if host is not None:
            connection_parameters["host"] = host

        if snowpark_session is None:
            snowpark_session = self._create_snowpark_session(
                connection_parameters
            )
        else:
            connection_parameters = (
                self._validate_snowpark_session_with_connection_parameters(
                    snowpark_session, connection_parameters
                )
            )

        self.snowpark_session: Session = snowpark_session
        self.connection_parameters: Dict[str, str] = connection_parameters
        self.use_staged_packages: bool = init_server_side_with_staged_packages
        self.use_account_event_table: bool = use_account_event_table

        if not is_otel_tracing_enabled() or not use_account_event_table:
            self._init_with_snowpark_session(
                snowpark_session,
                init_server_side,
                init_server_side_with_staged_packages,
                init_sis_dashboard,
                database_redact_keys,
                database_prefix,
                database_args,
                database_check_revision,
                connection_parameters,
            )

    def _create_snowpark_session(
        self, connection_parameters: Dict[str, Optional[str]]
    ):
        connection_parameters = connection_parameters.copy()
        # Validate.
        connection_parameters_to_set = []
        for k, v in connection_parameters.items():
            if v is None:
                connection_parameters_to_set.append(k)
        if connection_parameters_to_set:
            raise ValueError(
                f"If not supplying `snowpark_session` then must set `{connection_parameters_to_set}`!"
            )
        self.password_known = True
        # Create snowpark session making sure to create schema if it doesn't
        # already exist.
        schema = connection_parameters["schema"]
        del connection_parameters["schema"]
        snowpark_session = Session.builder.configs(
            connection_parameters
        ).create()
        self._validate_schema_name(schema)
        self._create_snowflake_schema_if_not_exists(snowpark_session, schema)
        return snowpark_session

    def _validate_snowpark_session_with_connection_parameters(
        self,
        snowpark_session: Session,
        connection_parameters: Dict[str, Optional[str]],
    ) -> Dict[str, str]:
        # Validate.
        snowpark_session_connection_parameters = {
            "account": snowpark_session.get_current_account(),
            "user": snowpark_session.get_current_user(),
            "database": snowpark_session.get_current_database(),
            "schema": snowpark_session.get_current_schema(),
            "warehouse": snowpark_session.get_current_warehouse(),
            "role": snowpark_session.get_current_role(),
        }

        for k, v in snowpark_session_connection_parameters.items():
            if v and v.startswith('"') and v.endswith('"'):
                snowpark_session_connection_parameters[k] = v.strip('"')

        missing_snowpark_session_parameters = []
        mismatched_parameters = []
        for k, v in snowpark_session_connection_parameters.items():
            if k in ["account", "user"] and not v:
                # Streamlit apps may hide these values so we don't check them.
                # They are required for a Snowpark Session anyway so this isn't
                # a problem (though we can't check consistency with
                # `connection_parameters`).
                continue
            if not v:
                missing_snowpark_session_parameters.append(k)
            elif connection_parameters[k] not in [None, v]:
                mismatched_parameters.append(k)
        if missing_snowpark_session_parameters:
            raise ValueError(
                f"Connection parameters missing from provided `snowpark_session`: {missing_snowpark_session_parameters}"
            )
        if mismatched_parameters:
            raise ValueError(
                f"Connection parameters mismatch between provided `snowpark_session` and args passed to `SnowflakeConnector`: {mismatched_parameters}"
            )
        # Check if password is also supplied as it's used in `run_dashboard`:
        # Passwords are inaccessible from the `snowpark_session` object and we
        # use another process to launch streamlit so must have the password on
        # hand.
        if connection_parameters["password"] is None:
            logger.warning(
                "Running the TruLens dashboard requires providing a `password` to the `SnowflakeConnector`."
            )
            snowpark_session_connection_parameters["password"] = "password"
            self.password_known = False
        else:
            snowpark_session_connection_parameters["password"] = (
                connection_parameters["password"]
            )
            self.password_known = True
        snowpark_session_connection_parameters = {
            k: v for k, v in snowpark_session_connection_parameters.items() if v
        }
        return snowpark_session_connection_parameters

    @staticmethod
    def _validate_snowpark_session_paramstyle(
        snowpark_session: Session,
    ) -> None:
        if snowpark_session.connection._paramstyle == "pyformat":
            # If this is the case, sql executions with bindings will fail later
            # on so we fail fast here.
            raise ValueError(
                "The Snowpark session must have paramstyle 'qmark'! To ensure"
                " this, during `snowflake.connector.connect` pass in"
                " `paramstyle='qmark'` or set"
                " `snowflake.connector.paramstyle = 'qmark'` beforehand."
            )

    def _init_with_snowpark_session(
        self,
        snowpark_session: Session,
        init_server_side: bool,
        init_server_side_with_staged_packages: bool,
        init_sis_dashboard: bool,
        database_redact_keys: bool,
        database_prefix: Optional[str],
        database_args: Optional[Dict[str, Any]],
        database_check_revision: bool,
        connection_parameters: Dict[str, str],
    ):
        self._validate_snowpark_session_paramstyle(snowpark_session)
        database_args = self._set_up_database_args(
            database_args,
            snowpark_session,
            connection_parameters,
            database_redact_keys,
            database_prefix,
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
                snowpark_session,
                connection_parameters["database"],
                connection_parameters["schema"],
                connection_parameters["warehouse"],
                database_args["database_prefix"],
                init_server_side_with_staged_packages,
            ).set_up_all()
        if init_sis_dashboard:
            self._set_up_sis_dashboard(
                session=snowpark_session,
                warehouse=connection_parameters["warehouse"],
                init_server_side_with_staged_packages=init_server_side_with_staged_packages,
            )

        # Add "trulens_workspace_version" tag to the current schema
        TRULENS_WORKSPACE_VERSION_TAG = "trulens_workspace_version"

        try:
            self._run_query(
                snowpark_session,
                f"CREATE TAG IF NOT EXISTS {TRULENS_WORKSPACE_VERSION_TAG}",
            )
            res = self._run_query(
                snowpark_session,
                "ALTER SCHEMA {}.{} SET TAG {}='{}'".format(
                    connection_parameters["database"],
                    connection_parameters["schema"],
                    TRULENS_WORKSPACE_VERSION_TAG,
                    self.db.get_db_revision(),
                ),
            )
            print(f"Set TruLens workspace version tag: {res}")
        except Exception as e:
            print(
                f"Error setting TruLens workspace version tag: {e}, check if you have enterprise version of Snowflake."
            )

    def _set_up_database_args(
        self,
        database_args: Dict[str, Any],
        snowpark_session: Session,
        connection_parameters: Dict[str, str],
        database_redact_keys: bool,
        database_prefix: Optional[str],
    ) -> Dict[str, Any]:
        database_args = database_args or {}
        # Set engine_params.
        default_engine_params = {
            "creator": lambda: snowpark_session.connection,
            "paramstyle": "qmark",
            # The following parameters ensure the pool does not allocate new
            # connections that it will close. This is a problem because the
            # "creator" does not create new connections, it only passes around
            # the single one it has.
            "max_overflow": 0,
            "pool_recycle": -1,
            "pool_timeout": 120,
        }
        if "engine_params" not in database_args:
            database_args["engine_params"] = default_engine_params
        else:
            for k, v in default_engine_params.items():
                if k in database_args["engine_params"]:
                    raise ValueError(
                        f"Cannot set `database_args['engine_params']['{k}']!"
                    )
        # Set remaining parameters.
        database_args.update({
            k: v
            for k, v in {
                "database_url": URL(**connection_parameters),
                "database_redact_keys": database_redact_keys,
            }.items()
            if v is not None
        })
        database_args["database_prefix"] = (
            database_prefix or core_db.DEFAULT_DATABASE_PREFIX
        )
        return database_args

    def _set_up_sis_dashboard(
        self,
        streamlit_name: str = "TRULENS_DASHBOARD",
        session: Optional[Session] = None,
        warehouse: Optional[str] = None,
        init_server_side_with_staged_packages: bool = False,
    ) -> None:
        return SiSDashboardArtifacts(
            streamlit_name,
            session or self.snowpark_session,
            self.connection_parameters["database"],
            self.connection_parameters["schema"],
            warehouse or self.connection_parameters["warehouse"],
            init_server_side_with_staged_packages or self.use_staged_packages,
        ).set_up_all()

    @staticmethod
    def _run_query(
        snowpark_session: Session,
        q: str,
        bindings: Optional[List[Any]] = None,
    ) -> List[Any]:
        cursor = snowpark_session.connection.cursor()
        if bindings:
            cursor.execute(q, bindings)
        else:
            cursor.execute(q)
        return cursor.fetchall()

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
        schema_name: str,
    ):
        SnowflakeConnector._run_query(
            snowpark_session,
            "CREATE SCHEMA IF NOT EXISTS IDENTIFIER(?)",
            [schema_name],
        )
        snowpark_session.use_schema(schema_name)

    @cached_property
    def db(self) -> DB:
        if isinstance(self._db, python_utils.OpaqueWrapper):
            self._db = self._db.unwrap()
        if not isinstance(self._db, DB):
            raise RuntimeError("Unhandled database type.")
        return self._db

    def initialize_snowflake_dao_fields(
        self,
        object_type: str,
        app_name: str,
        app_version: str,
    ) -> Tuple[ExternalAgentDao, RunDao, str, str]:
        snowflake_app_dao = None

        if not ObjectType.is_valid_object(object_type):
            raise ValueError(
                f"Invalid object_type to initialize Snowflake app: {object_type}"
            )

        if object_type == ObjectType.EXTERNAL_AGENT:
            logger.info(
                f"Initializing Snowflake External Agent DAO for app {app_name} version {app_version}"
            )
            # side effect: create external agent if not exist
            snowflake_app_dao = ExternalAgentDao(self.snowpark_session)
            snowflake_run_dao = RunDao(self.snowpark_session)
            agent_name, agent_version = (
                snowflake_app_dao.create_agent_if_not_exist(
                    name=app_name,
                    version=app_version,
                )
            )
            return (
                snowflake_app_dao,
                snowflake_run_dao,
                agent_name,
                agent_version,
            )

        raise ValueError(f"Object type {object_type} not supported.")
