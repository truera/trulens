"""
Tests for a Snowflake connection.
"""

from unittest import main
import uuid

from tests.test import optional_test
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeConnection(SnowflakeTestCase):
    @optional_test
    def test_basic_snowflake_connection(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_session("test_basic_snowflake_connection")

    @optional_test
    def test_connecting_to_premade_schema(self):
        # Create schema.
        schema_name = "test_connecting_to_premade_schema__"
        schema_name += str(uuid.uuid4()).replace("-", "_")
        schema_name = schema_name.upper()
        self.assertNotIn(schema_name, self.list_schemas())
        self._snowflake_schemas_to_delete.append(schema_name)
        self.run_query(f"CREATE SCHEMA {schema_name}")
        self.run_query(
            f"CREATE TABLE {self._database}.{schema_name}.MY_TABLE (TEST_COLUMN NUMBER)"
        )
        self.assertIn(schema_name, self.list_schemas())
        # Test that using this connection works.
        self.get_session(schema_name=schema_name, schema_already_exists=True)


if __name__ == "__main__":
    main()
