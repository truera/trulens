"""
Tests for a Snowflake connection.
"""

from unittest import main
import uuid

from trulens.dashboard import run_dashboard
from trulens.dashboard import stop_dashboard

from tests.test import optional_test
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeConnection(SnowflakeTestCase):
    @optional_test
    def test_snowflake_connection_via_snowpark_session(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_snowpark_session",
            connect_via_snowpark_session=True,
        )

    @optional_test
    def test_snowflake_connection_via_connection_parameters(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_connection_parameters",
            connect_via_snowpark_session=False,
        )

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

    @optional_test
    def test_run_leaderboard_without_password(self):
        session = self.get_session(
            "test_run_leaderboard_without_password",
            connect_via_snowpark_session=True,
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                "SnowflakeConnector was made via an established Snowpark session which did not pass through authentication details to the SnowflakeConnector. To fix, supply password argument during SnowflakeConnector construction.",
            ):
                run_dashboard(session)
        finally:
            # Clean up.
            try:
                stop_dashboard(session)
            except Exception:
                pass


if __name__ == "__main__":
    main()
