"""
Tests for a Snowflake connection.
"""

import os
import uuid

import pytest
import snowflake.connector
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector

from tests.util.snowflake_test_case import SnowflakeTestCase


@pytest.mark.snowflake
class TestSnowflakeConnection(SnowflakeTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_TRULENS_OTEL_TRACING = os.getenv("TRULENS_OTEL_TRACING")
        os.environ["TRULENS_OTEL_TRACING"] = "0"
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._orig_TRULENS_OTEL_TRACING is not None:
            os.environ["TRULENS_OTEL_TRACING"] = cls._orig_TRULENS_OTEL_TRACING
        else:
            del os.environ["TRULENS_OTEL_TRACING"]
        return super().tearDownClass()

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
