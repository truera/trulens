"""
Tests for SnowflakeConnector error handling, particularly for the '_db' attribute issue.
"""

import unittest
import unittest.mock

import pytest

try:
    # These imports require snowflake dependencies to be installed.
    from trulens.connectors.snowflake import SnowflakeConnector
except Exception:
    pass


@pytest.mark.snowflake
class TestSnowflakeConnectorErrorHandling(unittest.TestCase):
    def test_db_property_handles_missing_db_attribute(self):
        """Test that the db property raises a clear error when _db is not initialized."""
        # Create a mock SnowflakeConnector instance without proper initialization
        connector = object.__new__(SnowflakeConnector)
        # Don't set the _db attribute to simulate initialization failure

        with self.assertRaisesRegex(
            RuntimeError,
            "SnowflakeConnector was not properly initialized.*_db.*attribute is missing",
        ):
            _ = connector.db

    def test_db_property_handles_none_db_attribute(self):
        """Test that the db property raises a clear error when _db is None."""
        # Create a mock SnowflakeConnector instance with _db set to None
        connector = object.__new__(SnowflakeConnector)
        connector._db = None

        with self.assertRaisesRegex(
            RuntimeError,
            "SnowflakeConnector was not properly initialized.*_db.*attribute is missing or None",
        ):
            _ = connector.db

    @unittest.mock.patch(
        "trulens.connectors.snowflake.connector.SnowflakeEventTableDB"
    )
    @unittest.mock.patch(
        "trulens.connectors.snowflake.connector.is_otel_tracing_enabled"
    )
    def test_initialization_exception_handling(
        self, mock_otel_enabled, mock_event_table_db
    ):
        """Test that initialization exceptions are properly handled and logged."""
        # Setup mocks
        mock_otel_enabled.return_value = True
        mock_event_table_db.side_effect = Exception("Snowpark session error")

        # Mock snowpark session
        mock_session = unittest.mock.MagicMock()

        with self.assertLogs(level="ERROR") as log_context:
            with self.assertRaises(Exception):
                SnowflakeConnector(
                    snowpark_session=mock_session, use_account_event_table=True
                )

        # Verify that the error was logged
        self.assertTrue(
            any(
                "Failed to initialize SnowflakeEventTableDB"
                in record.getMessage()
                for record in log_context.records
            )
        )

    @unittest.mock.patch("trulens.connectors.snowflake.connector.logger")
    def test_get_events_error_handling_in_evaluator_context(self, mock_logger):
        """Test that get_events errors are handled gracefully in evaluator context."""
        # This test simulates what happens in the evaluator thread
        from trulens.core.utils.evaluator import Evaluator

        # Create a mock app with a connector that will fail
        mock_app = unittest.mock.MagicMock()
        mock_app.app_name = "test_app"
        mock_app.app_version = "v1"
        mock_app.connector.get_events.side_effect = RuntimeError(
            "SnowflakeConnector was not properly initialized"
        )

        evaluator = Evaluator(mock_app)

        # This should not raise an exception, but should return empty dict
        result = evaluator._get_record_id_to_unprocessed_events(None, None)

        self.assertEqual(result, {})

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        self.assertIn(
            "Failed to get events from connector",
            mock_logger.error.call_args[0][0],
        )
