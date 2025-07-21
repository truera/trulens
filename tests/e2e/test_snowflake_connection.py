"""
Tests for a Snowflake connection.
"""

from unittest import mock
import uuid

import pytest
import snowflake.connector
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.dashboard import run_dashboard
from trulens.dashboard import stop_dashboard

from tests.util.snowflake_test_case import SnowflakeTestCase


@pytest.mark.snowflake
class TestSnowflakeConnection(SnowflakeTestCase):
    def test_snowflake_connection_via_snowpark_session(self):
        """
        Check that we can connect to a Snowflake backend and have created the
        required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_snowpark_session",
            connect_via_snowpark_session=True,
        )

    def test_snowflake_connection_via_connection_parameters(self):
        """
        Check that we can connect to a Snowflake backend and have created the
        required schema.
        """
        self.get_session(
            "test_snowflake_connection_via_connection_parameters",
            connect_via_snowpark_session=False,
        )

    def test_connecting_to_premade_schema(self):
        # Create schema.
        schema_name = "test_connecting_to_premade_schema__"
        schema_name += str(uuid.uuid4()).replace("-", "_")
        schema_name = schema_name.upper()
        self.assertNotIn(schema_name, self.list_schemas())
        self._snowflake_schemas_to_delete.add(schema_name)
        self.run_query(f"CREATE SCHEMA {schema_name}")
        table_sql = (
            f"CREATE TABLE {self._database}.{schema_name}.MY_TABLE "
            f"(TEST_COLUMN NUMBER)"
        )
        self.run_query(table_sql)
        self.assertIn(schema_name, self.list_schemas())
        # Test that using this connection works.
        self.get_session(schema_name=schema_name, schema_already_exists=True)

    def test_run_leaderboard_without_password(self):
        session = self.get_session(
            "test_run_leaderboard_without_password",
            connect_via_snowpark_session=True,
        )

        # Mock subprocess.Popen to capture the arguments passed to streamlit
        with mock.patch("subprocess.Popen") as mock_popen:
            # Use a counter to control the mock behavior
            class MockStdout:
                def __init__(self):
                    self.call_count = 0
                    self.startup_sent = False

                def readline(self):
                    if not self.startup_sent:
                        self.startup_sent = True
                        return "Local URL: http://localhost:8501\n"
                    return ""  # Keep returning empty strings

            class MockStderr:
                def readline(self):
                    return ""  # Always return empty strings

            class MockProcess:
                def __init__(self):
                    self.stdout = MockStdout()
                    self.stderr = MockStderr()
                    self.poll_count = 0
                    self.terminated = False

                def poll(self):
                    if self.terminated:
                        return 1  # Process terminated
                    self.poll_count += 1
                    if self.poll_count > 10:  # Terminate after some calls
                        self.terminated = True
                        return 1
                    return None  # Process still running

            # Create instance of our mock process
            mock_process = MockProcess()

            mock_popen.return_value = mock_process

            try:
                run_dashboard(session)
                self.assertTrue(mock_popen.called)
                popen_args, _ = mock_popen.call_args
                streamlit_args = popen_args[0]

                # Check for external browser authentication
                self.assertIn("--snowflake-authenticator", streamlit_args)
                auth_index = streamlit_args.index("--snowflake-authenticator")
                self.assertEqual(
                    streamlit_args[auth_index + 1], "externalbrowser"
                )
            except Exception as e:
                self.fail(f"Dashboard startup failed: {e}")
            finally:
                # Clean up.
                try:
                    stop_dashboard(session)
                except Exception:
                    pass

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
