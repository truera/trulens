"""
Tests for a Snowflake connection.
"""

import uuid

import pytest
import snowflake.connector
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.dashboard import run_dashboard
from trulens.dashboard import stop_dashboard

from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeConnection(SnowflakeTestCase):
    @pytest.mark.optional
    def test_snowflake_connection_via_snowpark_session(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_snowpark_session",
            connect_via_snowpark_session=True,
        )

    @pytest.mark.optional
    def test_snowflake_connection_via_connection_parameters(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_connection_parameters",
            connect_via_snowpark_session=False,
        )

    @pytest.mark.optional
    def test_connecting_to_premade_schema(self):
        # Create schema.
        schema_name = "test_connecting_to_premade_schema__"
        schema_name += str(uuid.uuid4()).replace("-", "_")
        schema_name = schema_name.upper()
        self.assertNotIn(schema_name, self.list_schemas())
        self._snowflake_schemas_to_delete.add(schema_name)
        self.run_query(f"CREATE SCHEMA {schema_name}")
        self.run_query(
            f"CREATE TABLE {self._database}.{schema_name}.MY_TABLE (TEST_COLUMN NUMBER)"
        )
        self.assertIn(schema_name, self.list_schemas())
        # Test that using this connection works.
        self.get_session(schema_name=schema_name, schema_already_exists=True)

    @pytest.mark.optional
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

    @pytest.mark.optional
    def test_paramstyle_pyformat(self):
        default_paramstyle = snowflake.connector.paramstyle
        try:
            # pyformat paramstyle should fail fast.
            snowflake.connector.paramstyle = "pyformat"
            schema_name = self.create_and_use_schema(
                "test_paramstyle_pyformat", append_uuid=True
            )
            snowflake_connection = snowflake.connector.connect(
                **self._snowflake_connection_parameters, schema=schema_name
            )
            snowpark_session = Session.builder.configs({
                "connection": snowflake_connection
            }).create()
            with self.assertRaisesRegex(
                ValueError, "The Snowpark session must have paramstyle 'qmark'!"
            ):
                SnowflakeConnector(snowpark_session=snowpark_session)
            # qmark paramstyle should be fine.
            snowflake.connector.paramstyle = "qmark"
            snowflake_connection = snowflake.connector.connect(
                **self._snowflake_connection_parameters, schema=schema_name
            )
            snowpark_session = Session.builder.configs({
                "connection": snowflake_connection
            }).create()
            SnowflakeConnector(snowpark_session=snowpark_session)
        finally:
            snowflake.connector.paramstyle = default_paramstyle
