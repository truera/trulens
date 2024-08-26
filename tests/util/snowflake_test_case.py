"""
Test class to use for Snowflake testing.
"""

import logging
import os
from typing import Any, Dict
from unittest import TestCase
from unittest import main
import uuid

from snowflake.core import Root
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession


class SnowflakeTestCase(TestCase):
    def setUp(self):
        self._logger = logging.getLogger(__name__)
        self._snowflake_connection_parameters: Dict[str, Any] = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database_name": os.environ["SNOWFLAKE_DATABASE"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        self._snowflake_session = Session.builder.configs(
            self._snowflake_connection_parameters
        ).create()
        self._snowflake_root = Root(self._snowflake_session)
        self._snowflake_schemas_to_delete = []

    def tearDown(self):
        # [HACK!] Clean up any instances of `Tru` so tests don't interfere with each other.
        for key in [
            curr for curr in TruSession._instances if curr[0] == "TruSession"
        ]:
            del TruSession._instances[key]
        # Clean up any Snowflake schemas.
        schemas_not_deleted = []
        for curr in self._snowflake_schemas_to_delete:
            try:
                schema = self._snowflake_root.databases[
                    self._snowflake_connection_parameters["database_name"]
                ].schemas[curr]
                schema.delete()
            except Exception:
                schemas_not_deleted.append(curr)
                self._logger.error(f"Failed to clean up schema {curr}!")
        if schemas_not_deleted:
            error_msg = "Failed to clean up the following schemas:\n"
            error_msg += "\n".join(schemas_not_deleted)
            raise ValueError(error_msg)
        self._snowflake_session.close()

    def list_schemas(self):
        schemas = self._snowflake_root.databases[
            self._snowflake_connection_parameters["database_name"]
        ].schemas.iter()
        return [curr.name for curr in schemas]

    def get_session(self, app_base_name: str) -> TruSession:
        app_name = app_base_name
        app_name += "__"
        app_name += str(uuid.uuid4()).replace("-", "_")
        self._schema_name = app_name.upper()
        self.assertNotIn(self._schema_name, self.list_schemas())
        self._snowflake_schemas_to_delete.append(self._schema_name)
        connector = SnowflakeConnector(
            schema_name=self._schema_name,
            **self._snowflake_connection_parameters,
        )
        session = TruSession(connector=connector)
        self.assertIn(self._schema_name, self.list_schemas())
        return session


if __name__ == "__main__":
    main()
