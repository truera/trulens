"""
Tests for OTEL Evaluator.
"""

import time
from unittest.mock import MagicMock
from unittest.mock import patch
import weakref

import pandas as pd
import pytest
from trulens.core.utils.evaluator import Evaluator
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


@pytest.mark.optional
class TestOtelEvaluator(OtelTestCase):
    def test_evaluator_lifecycle(self):
        # Create a mock app.
        mock_app = MagicMock()
        mock_app.app_name = "Test App"
        mock_app.app_version = "v1"
        mock_app.app_id = "test_app_id"
        # Create evaluator.
        evaluator = Evaluator(mock_app)
        # Start evaluator.
        evaluator.start_evaluator()
        self.assertIsNotNone(evaluator._thread)
        self.assertTrue(evaluator._thread.is_alive())
        # Stop evaluator.
        evaluator_ref = weakref.ref(evaluator)
        thread_ref = weakref.ref(evaluator._thread)
        evaluator.stop_evaluator()
        time.sleep(2)
        self.assertIsNone(evaluator._thread)
        # Ensure everything is garbage collected.
        del evaluator
        self.assertCollected(evaluator_ref)
        self.assertCollected(thread_ref)

    def test_evaluator_processes_events(self):
        # Create a mock app.
        mock_app = MagicMock()
        mock_app.app_name = "App"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_id"
        # Mock the get_events method to return test data.
        test_events = pd.DataFrame([
            {
                "trace": {"span_id": "span1", "parent_id": None},
                "record_attributes": {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.RECORD_ID: "test_record_id",
                },
            },
            {
                "trace": {"span_id": "span2", "parent_id": "span1"},
                "record_attributes": {
                    SpanAttributes.RECORD_ID: "test_record_id"
                },
            },
        ])
        # Mock the connector to return test events.
        mock_app.connector.get_events.return_value = test_events
        evaluator = Evaluator(mock_app)
        # Start evaluator.
        evaluator.start_evaluator()
        time.sleep(1)
        # Verify compute_feedbacks was called correctly.
        mock_app.compute_feedbacks.assert_called_once()
        pd.testing.assert_frame_equal(
            test_events,
            mock_app.compute_feedbacks.call_args_list[0].kwargs["events"],
        )
        # Stop evaluator
        evaluator.stop_evaluator()

    def test_evaluator_error_handling(self):
        """Test that evaluator handles errors gracefully."""
        # Create a mock app that raises an error when compute_feedbacks is
        # called.
        mock_app = MagicMock()
        mock_app.app_name = "Error App"
        mock_app.app_version = "v1"
        mock_app.app_id = "error_app_id"
        mock_app.compute_feedbacks.side_effect = Exception("Test error")
        # Create test data
        test_events = pd.DataFrame([
            {
                "trace": {"span_id": "span1", "parent_id": None},
                "record_attributes": {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.RECORD_ID: "test_record_id",
                },
            }
        ])
        # Mock the connector to return test events.
        mock_app.connector.get_events.return_value = test_events
        # Start evaluator.
        evaluator = Evaluator(mock_app)
        evaluator.start_evaluator()
        time.sleep(1)
        # Verify compute_feedbacks was called even though it raised an error.
        mock_app.compute_feedbacks.assert_called()

    def test_evaluator_invalid_start(self):
        mock_app = MagicMock()
        mock_app.app_name = "Test App"
        mock_app.app_version = "v1"
        evaluator = Evaluator(mock_app)
        # Try starting evaluator twice.
        evaluator.start_evaluator()
        with self.assertRaises(RuntimeError):
            evaluator.start_evaluator()
        evaluator.stop_evaluator()
        # Try starting evaluator without OTEL tracing.
        with patch(
            "trulens.core.utils.evaluator.is_otel_tracing_enabled",
            return_value=False,
        ):
            with self.assertRaises(ValueError):
                evaluator.start_evaluator()
