"""
Tests for a Snowflake connection.
"""

import os
from unittest import TestCase
from unittest import main
import uuid

from snowflake.core import Root
from snowflake.snowpark import Session
from trulens.core import Tru

from tests.unit.utils import optional_test


class TestSnowflakeConnection(TestCase):
    def setUp(self):
        self._snowflake_connection_parameters = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        self._snowflake_session = Session.builder.configs(
            self._snowflake_connection_parameters
        ).create()
        self._snowflake_root = Root(self._snowflake_session)

    def tearDown(self):
        self._snowflake_session.close()

    def _list_schemas(self):
        schemas = self._snowflake_root.databases[
            self._snowflake_connection_parameters["database"]
        ].schemas.iter()
        return [curr.name for curr in schemas]

    @optional_test
    def test_basic_snowflake_connection(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        app_name = str(uuid.uuid4()).replace("-", "_")
        schema_name = Tru._validate_and_compute_schema_name(app_name)
        try:
            self.assertNotIn(schema_name, self._list_schemas())
            Tru(
                snowflake_connection_parameters=self._snowflake_connection_parameters,
                name=app_name,
            )
            self.assertIn(schema_name, self._list_schemas())
        finally:
            if schema_name in self._list_schemas():
                schema = self._snowflake_root.databases[
                    self._snowflake_connection_parameters["database"]
                ].schemas[schema_name]
                schema.delete()


if __name__ == "__main__":
    main()
