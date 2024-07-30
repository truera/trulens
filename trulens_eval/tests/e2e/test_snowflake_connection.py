"""
Tests for a Snowflake connection. 
"""

import os
from unittest import main
from unittest import TestCase
import uuid

from snowflake.core import Root
from snowflake.snowpark import Session
from tests.unit.test import optional_test

from trulens_eval import Tru


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
            self._snowflake_connection_parameters["database"]].schemas.iter()
        return [curr.name for curr in schemas]

    @optional_test
    def test_basic_snowflake_connection(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        app_name = str(uuid.uuid4()).replace("-", "_").upper()
        try:
            self.assertNotIn(app_name, self._list_schemas())
            tru = Tru(
                snowflake_connection_parameters=self.
                _snowflake_connection_parameters,
                name=app_name
            )
            self.assertIn(app_name, self._list_schemas())
        finally:
            if app_name in self._list_schemas():
                schema = self._snowflake_root.databases[
                    self._snowflake_connection_parameters["database"]
                ].schemas[app_name]
                schema.delete()


if __name__ == '__main__':
    main()
