"""
Test class to use for Snowflake testing.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from unittest import TestCase
import uuid

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors import snowflake as snowflake_connector
from trulens.core import session as core_session
from trulens.providers.cortex.provider import Cortex


class SnowflakeTestCase(TestCase):
    def setUp(self):
        self._logger = logging.getLogger(__name__)
        self._database = os.environ["SNOWFLAKE_DATABASE"]
        self._snowflake_connection_parameters: Dict[str, str] = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        self._snowpark_session = Session.builder.configs(
            self._snowflake_connection_parameters
        ).create()
        self._snowflake_schemas_to_delete = set()
        Cortex.DEFAULT_SNOWPARK_SESSION = self._snowpark_session

    def tearDown(self):
        # [HACK!] Clean up any instances of `TruSession` so tests don't interfere with each other.
        for key in [
            curr
            for curr in core_session.TruSession._singleton_instances
            if curr[0] == "trulens.core.session.TruSession"
        ]:
            del core_session.TruSession._singleton_instances[key]
        # Clean up any Snowflake schemas.
        schemas_not_deleted = []
        for curr in self._snowflake_schemas_to_delete:
            try:
                self.run_query(
                    f"DROP SCHEMA {self._snowflake_connection_parameters['database']}.{curr}"
                )
            except Exception:
                schemas_not_deleted.append(curr)
                self._logger.error(f"Failed to clean up schema {curr}!")
        # Check if any artifacts weren't deleted.
        if schemas_not_deleted:
            error_msg = "Failed to clean up the following schemas:\n"
            error_msg += "\n".join(schemas_not_deleted)
            raise ValueError(error_msg)
        # Close session.
        self._snowpark_session.close()

    def list_schemas(self):
        res = self.run_query(
            f"SHOW SCHEMAS IN DATABASE {self._snowflake_connection_parameters['database']}"
        )
        return [curr["name"] for curr in res]

    def get_snowpark_session_with_schema(self, schema: str) -> Session:
        snowflake_connection_parameters = (
            self._snowflake_connection_parameters.copy()
        )
        snowflake_connection_parameters["schema"] = schema
        return Session.builder.configs(snowflake_connection_parameters).create()

    def get_session(
        self,
        app_base_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        schema_already_exists: bool = False,
        connect_via_snowpark_session: bool = True,
    ) -> core_session.TruSession:
        if bool(app_base_name) == bool(schema_name):
            raise ValueError(
                "Exactly one of `app_base_name` and `schema_name` must be supplied!"
            )
        if app_base_name:
            app_name = app_base_name
            app_name += "__"
            app_name += str(uuid.uuid4()).replace("-", "_")
            self._schema = app_name
        else:
            self._schema = schema_name
        self._schema = self._schema.upper()
        if not schema_already_exists:
            self.assertNotIn(self._schema, self.list_schemas())
            self._snowflake_schemas_to_delete.add(self._schema)
        if not connect_via_snowpark_session:
            connector = snowflake_connector.SnowflakeConnector(
                schema=self._schema,
                **self._snowflake_connection_parameters,
                init_server_side=True,
                init_server_side_with_staged_packages=True,
            )
        else:
            if not schema_already_exists:
                self.create_and_use_schema(self._schema)
            connector = snowflake_connector.SnowflakeConnector(
                snowpark_session=self._snowpark_session,
                init_server_side=True,
                init_server_side_with_staged_packages=True,
            )
        session = core_session.TruSession(connector=connector)
        self.assertIn(self._schema, self.list_schemas())
        return session

    def run_query(
        self, q: str, bindings: Optional[List[Any]] = None
    ) -> List[Row]:
        return self._snowpark_session.sql(q, bindings).collect()

    def create_and_use_schema(
        self,
        schema_name: str,
        append_uuid: bool = False,
        delete_schema_on_cleanup: bool = True,
    ) -> str:
        schema_name = schema_name.upper()
        if append_uuid:
            schema_name = (
                f"{schema_name}__{str(uuid.uuid4()).replace('-', '_')}"
            )
        self._schema = schema_name
        self.run_query(
            "CREATE SCHEMA IF NOT EXISTS IDENTIFIER(?)", [schema_name]
        )
        if delete_schema_on_cleanup:
            self._snowflake_schemas_to_delete.add(schema_name)
        self._snowpark_session.use_schema(schema_name)
        return schema_name
