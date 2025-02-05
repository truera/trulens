"""
Tests for OTEL Snowflake Exporter.
"""

import unittest
import unittest.mock

from trulens.connectors.snowflake import SnowflakeConnector
from trulens.connectors.snowflake.otel_exporter import (
    TruLensSnowflakeSpanExporter,
)


class TestOtelSnowflakeEventTableExporter(unittest.TestCase):
    def test_dry_run_success(self) -> None:
        # Mock SnowflakeConnector.
        mock_connector = unittest.mock.MagicMock()
        mock_connector.__class__ = SnowflakeConnector
        # Initialize exporter with mock connector.
        TruLensSnowflakeSpanExporter(
            connector=mock_connector, verify_via_dry_run=True
        )
        # Verify that the SQL commands were called.
        mock_connector.snowpark_session.sql.assert_any_call("SELECT 20240131")
        mock_connector.snowpark_session.sql.assert_any_call(
            "CREATE TEMP STAGE IF NOT EXISTS trulens_spans"
        )
        mock_connector.snowpark_session.file.put.assert_called_once()

    def test_dry_run_failure(self) -> None:
        # Mock SnowflakeConnector.
        mock_connector = unittest.mock.MagicMock()
        mock_connector.__class__ = SnowflakeConnector
        mock_connector.snowpark_session.file.put.side_effect = ValueError(
            "Error while putting to stage!"
        )
        # Initialize exporter with mock connector.
        with self.assertRaisesRegex(
            ValueError, "Error while putting to stage!"
        ):
            TruLensSnowflakeSpanExporter(
                connector=mock_connector, verify_via_dry_run=True
            )


if __name__ == "__main__":
    unittest.main()
