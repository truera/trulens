"""
Tests for OTEL wait_for_records, wait_for_feedback_results, and retrieve_feedback_results.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core import Feedback
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.core.utils.evaluator import Evaluator
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


@pytest.mark.optional
class TestOtelWaitForRecords(OtelTestCase):
    """Tests for the OTEL-specific wait_for_records implementation."""

    def test_wait_for_records_finds_record_root(self):
        """Test that wait_for_records finds RECORD_ROOT spans."""

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "test answer"

        app = _TestApp()
        tru_app = TruApp(
            app, app_name="test_wait_for_records", app_version="v1"
        )

        with tru_app as recording:
            app.query("test question")

        record_ids = [rec.record_id for rec in recording.records]
        self.assertEqual(1, len(record_ids))

        # Should not raise - record should be found
        tru_session = TruSession()
        tru_session.wait_for_records(
            record_ids=record_ids, timeout=10, poll_interval=0.1
        )

    def test_wait_for_records_multiple_records(self):
        """Test that wait_for_records can wait for multiple records."""

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return f"answer to {question}"

        app = _TestApp()
        tru_app = TruApp(app, app_name="test_wait_multiple", app_version="v1")

        record_ids = []
        with tru_app as recording1:
            app.query("question 1")
            record_ids.extend([rec.record_id for rec in recording1.records])

        with tru_app as recording2:
            app.query("question 2")
            record_ids.extend([rec.record_id for rec in recording2.records])

        self.assertEqual(2, len(record_ids))

        # Should not raise - both records should be found
        tru_session = TruSession()
        tru_session.wait_for_records(
            record_ids=record_ids, timeout=10, poll_interval=0.1
        )

    def test_wait_for_records_timeout(self):
        """Test that wait_for_records raises RuntimeError on timeout."""
        tru_session = TruSession()

        # Wait for a non-existent record should timeout
        with self.assertRaises(RuntimeError) as context:
            tru_session.wait_for_records(
                record_ids=["non_existent_record_id"],
                timeout=1,
                poll_interval=0.1,
            )

        self.assertIn("non_existent_record_id", str(context.exception))

    def test_wait_for_records_no_start_time_filter(self):
        """Test that wait_for_records doesn't apply start_time filter."""

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "answer"

        app = _TestApp()
        tru_app = TruApp(app, app_name="test_no_start_time", app_version="v1")

        with tru_app as recording:
            app.query("test")

        record_ids = [rec.record_id for rec in recording.records]

        # Mock get_events to verify start_time is None
        tru_session = TruSession()
        original_get_events = tru_session.connector.get_events

        captured_calls = []

        def mock_get_events(*args, **kwargs):
            captured_calls.append(kwargs)
            return original_get_events(*args, **kwargs)

        with patch.object(
            tru_session.connector, "get_events", side_effect=mock_get_events
        ):
            tru_session.wait_for_records(
                record_ids=record_ids, timeout=10, poll_interval=0.1
            )

        # Verify start_time was None in all calls
        for call in captured_calls:
            self.assertIsNone(call.get("start_time"))


@pytest.mark.optional
class TestOtelWaitForFeedbackResults(OtelTestCase):
    """Tests for the wait_for_feedback_results method."""

    def test_wait_for_feedback_results_finds_eval_roots(self):
        """Test that wait_for_feedback_results finds EVAL_ROOT spans."""

        # Create mock feedback provider
        def mock_feedback_fn(text: str) -> float:
            return 0.8

        mock_provider = MagicMock()
        mock_provider.mock_feedback = mock_feedback_fn

        f_mock = Feedback(
            mock_provider.mock_feedback, name="MockFeedback"
        ).on_input()

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "test answer"

        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_wait_feedback",
            app_version="v1",
            feedbacks=[f_mock],
        )

        with tru_app as recording:
            app.query("test question")

        record_ids = [rec.record_id for rec in recording.records]
        tru_session = TruSession()

        # Wait for records first
        tru_session.wait_for_records(record_ids=record_ids, timeout=10)

        # Compute feedbacks
        tru_app._evaluator.compute_now(record_ids)

        # Wait for feedback results
        tru_session.wait_for_feedback_results(
            record_ids=record_ids,
            feedback_names=["MockFeedback"],
            timeout=30,
            poll_interval=0.1,
        )

        # Verify EVAL_ROOT spans exist
        events = tru_session.connector.get_events(
            app_name=None,
            app_version=None,
            record_ids=record_ids,
            start_time=None,
        )

        eval_root_count = 0
        for _, event in events.iterrows():
            record_attrs = event.get("record_attributes", {})
            if isinstance(record_attrs, str):
                import json

                record_attrs = json.loads(record_attrs)
            if (
                record_attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.EVAL_ROOT
            ):
                eval_root_count += 1

        self.assertGreater(eval_root_count, 0)

    def test_wait_for_feedback_results_timeout(self):
        """Test that wait_for_feedback_results raises RuntimeError on timeout."""
        tru_session = TruSession()

        # Create a record first
        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "answer"

        app = _TestApp()
        tru_app = TruApp(
            app, app_name="test_feedback_timeout", app_version="v1"
        )

        with tru_app as recording:
            app.query("test")

        record_ids = [rec.record_id for rec in recording.records]
        tru_session.wait_for_records(record_ids=record_ids, timeout=10)

        # Wait for feedback results that don't exist should timeout
        with self.assertRaises(RuntimeError) as context:
            tru_session.wait_for_feedback_results(
                record_ids=record_ids,
                feedback_names=["NonExistentFeedback"],
                timeout=1,
                poll_interval=0.1,
            )

        self.assertIn("Missing", str(context.exception))

    def test_wait_for_feedback_results_legacy_mode_no_op(self):
        """Test that wait_for_feedback_results is a no-op in legacy mode."""
        with patch(
            "trulens.core.session.is_otel_tracing_enabled",
            return_value=False,
        ):
            tru_session = TruSession()
            # Should not raise even with invalid inputs in legacy mode
            tru_session.wait_for_feedback_results(
                record_ids=["any"],
                feedback_names=["any"],
                timeout=1,
            )


@pytest.mark.optional
class TestOtelRetrieveFeedbackResults(OtelTestCase):
    """Tests for the retrieve_feedback_results method."""

    def test_retrieve_feedback_results_returns_dataframe(self):
        """Test that retrieve_feedback_results returns a DataFrame with feedback columns."""

        def mock_feedback_fn(text: str) -> float:
            return 0.75

        mock_provider = MagicMock()
        mock_provider.mock_feedback = mock_feedback_fn

        f_mock = Feedback(
            mock_provider.mock_feedback, name="TestFeedback"
        ).on_input()

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "test answer"

        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_retrieve",
            app_version="v1",
            feedbacks=[f_mock],
        )

        with tru_app as recording:
            app.query("test question")

        # Retrieve feedback results
        results_df = recording.retrieve_feedback_results(timeout=60)

        # Verify DataFrame structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(1, len(results_df))
        self.assertIn("TestFeedback", results_df.columns)

    def test_retrieve_feedback_results_multiple_records(self):
        """Test retrieve_feedback_results with multiple records in same recording."""

        def mock_feedback_fn(text: str) -> float:
            return 0.9

        mock_provider = MagicMock()
        mock_provider.mock_feedback = mock_feedback_fn

        f_mock = Feedback(
            mock_provider.mock_feedback, name="MultiFeedback"
        ).on_input()

        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return f"answer to {question}"

        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_retrieve_multi",
            app_version="v1",
            feedbacks=[f_mock],
        )

        # Create multiple records
        with tru_app as recording:
            app.query("question 1")
            app.query("question 2")

        # Retrieve feedback results
        results_df = recording.retrieve_feedback_results(timeout=60)

        # Should have results for multiple records
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(recording.records), len(results_df))


@pytest.mark.optional
class TestEvaluatorNoStartTimeFilter(OtelTestCase):
    """Tests for the fix that removes start_time filter when record_ids are provided."""

    def test_get_record_id_to_unprocessed_events_no_start_time_when_record_ids(
        self,
    ):
        """Test that start_time filter is not applied when record_ids are specified."""
        mock_app = MagicMock()
        mock_app.app_name = "Test App"
        mock_app.app_version = "v1"

        evaluator = Evaluator(mock_app)

        # Create test events
        test_events = pd.DataFrame([
            {
                "trace": {"span_id": "span1", "parent_id": None},
                "record_attributes": {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.RECORD_ID: "test_record_id",
                },
            },
        ])

        captured_calls = []

        def mock_get_events(*args, **kwargs):
            captured_calls.append(kwargs)
            return test_events

        mock_app.connector.get_events = mock_get_events

        # Set a processed_time to verify it's NOT used when record_ids is provided
        import datetime

        evaluator._processed_time = datetime.datetime.now()

        # Call with specific record_ids
        evaluator._get_record_id_to_unprocessed_events(
            record_ids=["test_record_id"],
            start_time=evaluator._processed_time,
        )

        # Verify start_time was None (not the processed_time)
        self.assertEqual(1, len(captured_calls))
        self.assertIsNone(captured_calls[0].get("start_time"))

    def test_get_record_id_to_unprocessed_events_uses_start_time_when_no_record_ids(
        self,
    ):
        """Test that start_time filter IS applied when record_ids is None."""
        mock_app = MagicMock()
        mock_app.app_name = "Test App"
        mock_app.app_version = "v1"

        evaluator = Evaluator(mock_app)

        # Create test events
        test_events = pd.DataFrame([
            {
                "trace": {"span_id": "span1", "parent_id": None},
                "record_attributes": {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.RECORD_ID: "test_record_id",
                },
            },
        ])

        captured_calls = []

        def mock_get_events(*args, **kwargs):
            captured_calls.append(kwargs)
            return test_events

        mock_app.connector.get_events = mock_get_events

        # Set a processed_time
        import datetime

        test_start_time = datetime.datetime.now()
        evaluator._processed_time = test_start_time

        # Call without specific record_ids (None)
        evaluator._get_record_id_to_unprocessed_events(
            record_ids=None,
            start_time=test_start_time,
        )

        # Verify start_time WAS used
        self.assertEqual(1, len(captured_calls))
        self.assertEqual(test_start_time, captured_calls[0].get("start_time"))
