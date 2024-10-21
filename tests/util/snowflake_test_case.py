"""
Test class to use for Snowflake testing.
"""

import logging
import os
from typing import Dict, Optional
from unittest import TestCase
from unittest import main
import uuid

from snowflake.core import Root
from snowflake.snowpark import Session
from trulens.connectors import snowflake as snowflake_connector
from trulens.core import session as core_session


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
        self._snowflake_root = Root(self._snowpark_session)
        self._snowflake_schemas_to_delete = []

    def tearDown(self):
        # [HACK!] Clean up any instances of `TruSession` so tests don't interfere with each other.
        for key in [
            curr
            for curr in core_session.TruSession._instances
            if curr[0] == "TruSession"
        ]:
            del core_session.TruSession._instances[key]
        # Clean up any Snowflake schemas.
        schemas_not_deleted = []
        for curr in self._snowflake_schemas_to_delete:
            try:
                schema = self._snowflake_root.databases[
                    self._snowflake_connection_parameters["database"]
                ].schemas[curr]
                schema.delete()
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
        schemas = self._snowflake_root.databases[
            self._snowflake_connection_parameters["database"]
        ].schemas.iter()
        return [curr.name for curr in schemas]

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
            self._snowflake_schemas_to_delete.append(self._schema)
        if not connect_via_snowpark_session:
            connector = snowflake_connector.SnowflakeConnector(
                schema=self._schema,
                **self._snowflake_connection_parameters,
                init_server_side=True,
            )
        else:
            if not schema_already_exists:
                snowflake_connector.SnowflakeConnector._create_snowflake_schema_if_not_exists(
                    self._snowpark_session, self._database, self._schema
                )
            connector = snowflake_connector.SnowflakeConnector(
                snowpark_session=self._snowpark_session,
                init_server_side=True,
            )
        session = core_session.TruSession(connector=connector)
        self.assertIn(self._schema, self.list_schemas())
        return session

    def run_query(self, q: str) -> None:
        self._snowpark_session.sql(q).collect()


if __name__ == "__main__":
    main()
