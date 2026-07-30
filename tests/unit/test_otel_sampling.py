"""
Integration tests for sampling in OTEL mode.

Validates that:
- configure_online_eval() stores the controller on TruSession
- The evaluator respects sampling in the ingest path
- The batch path (compute_now) is NOT gated by sampling
- EVAL_DECISION spans carry the expected attributes
- sampled / sample_rate / eval_decision_reason columns round-trip
  through get_records_and_feedback
"""

import time
from typing import List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.core.utils.evaluator import Evaluator
from trulens.core.utils.evaluator import _emit_sampling_decision_span
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


def _make_mock_events(record_id: str = "test_record_id") -> pd.DataFrame:
    """Create minimal events DataFrame for testing."""
    return pd.DataFrame([
        {
            "trace": {"span_id": "span1", "parent_id": None},
            "record_attributes": {
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.RECORD_ID: record_id,
            },
            "resource_attributes": {
                "ai.observability.app_name": "test_app",
                "ai.observability.app_version": "v1",
                "ai.observability.app_id": "app_hash_test",
            },
        },
    ])


@pytest.mark.optional
class TestConfigureOnlineEval(OtelTestCase):
    """Tests for TruSession.configure_online_eval()."""

    def test_stores_controller(self):
        session = TruSession()
        self.assertIsNone(session.sampling_controller)
        session.configure_online_eval(sample_rate=0.5, throttle=100)
        self.assertIsNotNone(session.sampling_controller)
        self.assertEqual(session.sampling_controller.config.sample_rate, 0.5)
        self.assertEqual(session.sampling_controller.config.throttle, 100)

    def test_second_call_replaces(self):
        session = TruSession()
        session.configure_online_eval(sample_rate=0.1)
        session.configure_online_eval(sample_rate=0.9)
        self.assertEqual(session.sampling_controller.config.sample_rate, 0.9)

    def test_per_app_rates(self):
        session = TruSession()
        session.configure_online_eval(sample_rate={"app_a": 0.1, "app_b": 0.9})
        self.assertEqual(
            session.sampling_controller.config.sample_rate,
            {"app_a": 0.1, "app_b": 0.9},
        )


@pytest.mark.optional
class TestEvaluatorSamplingGate(OtelTestCase):
    """Tests that sampling gates the evaluator correctly."""

    def _make_evaluator(self):
        mock_app = MagicMock()
        mock_app.app_name = "test_app"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_hash_test"
        mock_app.connector = MagicMock()
        return Evaluator(mock_app), mock_app

    def test_sampling_skips_records_in_evaluator_thread(self):
        """With sample_rate=0.0, the evaluator skips all records."""
        evaluator, mock_app = self._make_evaluator()

        # Configure sampling to skip everything.
        session = TruSession()
        session.configure_online_eval(sample_rate=0.0)

        events = _make_mock_events()
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"test_record_id": events}
        )

        # Run in evaluator thread mode (in_evaluator_thread=True).
        evaluator._compute_feedbacks(in_evaluator_thread=True)

        # compute_feedbacks should NOT have been called on the app.
        mock_app.compute_feedbacks.assert_not_called()

    def test_batch_path_ignores_sampling(self):
        """compute_now (in_evaluator_thread=False) evaluates everything."""
        evaluator, mock_app = self._make_evaluator()

        # Configure sampling to skip everything.
        session = TruSession()
        session.configure_online_eval(sample_rate=0.0)

        events = _make_mock_events()
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"test_record_id": events}
        )

        # Run via compute_now (in_evaluator_thread=False).
        evaluator._compute_feedbacks(in_evaluator_thread=False)

        # compute_feedbacks SHOULD have been called because batch path
        # is exempt from sampling.
        mock_app.compute_feedbacks.assert_called_once()

    def test_no_sampling_configured_evaluates_all(self):
        """Without configure_online_eval(), all records are evaluated."""
        evaluator, mock_app = self._make_evaluator()

        # Ensure no sampling controller is set.
        session = TruSession()
        session._sampling_controller = None

        events = _make_mock_events()
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"test_record_id": events}
        )

        evaluator._compute_feedbacks(in_evaluator_thread=True)

        mock_app.compute_feedbacks.assert_called_once()


@pytest.mark.optional
class TestEvalDecisionSpanType(OtelTestCase):
    """Test that EVAL_DECISION span type and attributes exist."""

    def test_span_type_exists(self):
        self.assertEqual(
            SpanAttributes.SpanType.EVAL_DECISION.value, "eval_decision"
        )

    def test_attributes_defined(self):
        self.assertTrue(hasattr(SpanAttributes.EVAL_DECISION, "SAMPLE_RATE"))
        self.assertTrue(
            hasattr(SpanAttributes.EVAL_DECISION, "EVAL_DECISION_REASON")
        )


@pytest.mark.optional
class TestSkipPathEmitsDecisionSpans(OtelTestCase):
    """Verify that skipped records also get EVAL_DECISION spans."""

    def _make_evaluator(self):
        mock_app = MagicMock()
        mock_app.app_name = "test_app"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_hash_test"
        mock_app.connector = MagicMock()
        return Evaluator(mock_app), mock_app

    def test_skip_path_emits_decision_span(self):
        """With sample_rate=0.0, records are skipped but decision spans
        are still emitted (so coverage is measurable)."""
        evaluator, mock_app = self._make_evaluator()

        session = TruSession()
        session.configure_online_eval(sample_rate=0.0)

        # Create 3 records.
        events_map = {
            f"record_{i}": _make_mock_events(f"record_{i}") for i in range(3)
        }
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value=events_map
        )

        evaluator._compute_feedbacks(in_evaluator_thread=True)

        # compute_feedbacks should NOT have been called (all skipped).
        mock_app.compute_feedbacks.assert_not_called()

        # But the controller's counters should show 3 not_sampled.
        counters = session.sampling_controller.counters
        self.assertEqual(counters["not_sampled"], 3)
        self.assertEqual(counters["evaluated"], 0)


@pytest.mark.optional
class TestSampledOutRecordsReachableViaComputeNow(OtelTestCase):
    """Records skipped by sampling must remain reachable for explicit backfill.

    This is the regression test for the state-modeling bug where
    sampled-out records were marked as processed in
    _record_id_to_event_count, making them invisible to compute_now().
    """

    def _make_evaluator(self):
        mock_app = MagicMock()
        mock_app.app_name = "test_app"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_hash_test"
        mock_app.connector = MagicMock()
        return Evaluator(mock_app), mock_app

    def test_compute_now_reaches_sampled_out_records(self):
        evaluator, mock_app = self._make_evaluator()

        session = TruSession()
        session.configure_online_eval(sample_rate=0.0)

        record_ids = ["record_A", "record_B", "record_C"]
        events_map = {rid: _make_mock_events(rid) for rid in record_ids}

        # Step 1: automatic evaluator skips all records.
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value=events_map
        )
        evaluator._compute_feedbacks(in_evaluator_thread=True)
        mock_app.compute_feedbacks.assert_not_called()

        # Verify they're tracked as sampled-out, NOT as processed.
        for rid in record_ids:
            self.assertIn(rid, evaluator._sampled_out_record_ids)
            self.assertNotIn(rid, evaluator._record_id_to_event_count)

        # Step 2: explicit backfill via compute_now should reach them.
        # Reset the mock so we get a fresh return value for the
        # force-path fetch.
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value=events_map
        )
        evaluator.compute_now(record_ids=record_ids)

        # compute_feedbacks should have been called for each record.
        self.assertEqual(mock_app.compute_feedbacks.call_count, 3)

    def test_automatic_evaluator_does_not_reprocess_sampled_out(self):
        """The automatic path should not re-consider sampled-out records
        on every poll cycle."""
        evaluator, mock_app = self._make_evaluator()

        session = TruSession()
        session.configure_online_eval(sample_rate=0.0)

        events_map = {"record_X": _make_mock_events("record_X")}
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value=events_map
        )

        # First automatic pass: record is sampled-out.
        evaluator._compute_feedbacks(in_evaluator_thread=True)
        mock_app.compute_feedbacks.assert_not_called()

        # Second automatic pass: _get_record_id_to_unprocessed_events
        # is called again but the record should be filtered out by the
        # _sampled_out_record_ids check.  We test this by NOT mocking
        # the method and letting the real implementation run — it will
        # return an empty dict because the record is in
        # _sampled_out_record_ids.
        evaluator._get_record_id_to_unprocessed_events = (
            evaluator.__class__._get_record_id_to_unprocessed_events.__get__(
                evaluator
            )
        )
        # The connector returns the same events.
        mock_app.connector.get_events.return_value = pd.DataFrame([
            {
                "trace": {"span_id": "span1", "parent_id": None},
                "record_attributes": {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.RECORD_ID: "record_X",
                },
            },
        ])
        evaluator._compute_feedbacks(in_evaluator_thread=True)
        # Still not called — the record was filtered out.
        mock_app.compute_feedbacks.assert_not_called()


@pytest.mark.optional
class TestOutOfScopeAppStillEvaluatesOtel(OtelTestCase):
    """OTEL integration: per-app sampling must not disable other apps."""

    def _make_evaluator(self):
        mock_app = MagicMock()
        mock_app.app_name = "other_app"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_hash_other"
        mock_app.connector = MagicMock()
        return Evaluator(mock_app), mock_app

    def test_out_of_scope_app_evaluates(self):
        evaluator, mock_app = self._make_evaluator()

        session = TruSession()
        # Only scope sampling to "prod_rag", not "other_app".
        session.configure_online_eval(sample_rate={"prod_rag": 0.1})

        events = _make_mock_events("record_1")
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"record_1": events}
        )

        evaluator._compute_feedbacks(in_evaluator_thread=True)

        # other_app should still be evaluated.
        mock_app.compute_feedbacks.assert_called_once()


@pytest.mark.optional
class TestSamplingControllerInjectable(OtelTestCase):
    """Verify the controller setter works for test injection."""

    def test_setter_injects_controller(self):
        from trulens.core.sampling import SamplingConfig
        from trulens.core.sampling import SamplingController

        session = TruSession()
        ctrl = SamplingController(SamplingConfig(sample_rate=0.0))
        session.sampling_controller = ctrl
        self.assertIs(session.sampling_controller, ctrl)


@pytest.mark.optional
class TestSamplingDecisionProjection(OtelTestCase):
    """End-to-end: EVAL_DECISION spans round-trip through the DB and
    get_records_and_feedback projects sampled/sample_rate/reason columns."""

    def test_evaluated_and_skipped_records_in_dataframe(self):
        """Log records, emit decision spans, verify the DataFrame has
        both True and False in ``sampled`` with correct metadata."""

        class _App:
            @instrument()
            def run(self, x: str) -> str:
                return f"echo {x}"

        app = _App()
        tru_app = TruApp(
            app,
            app_name="projection_test",
            app_version="v1",
        )

        # Log 2 records.
        record_ids: List[str] = []
        for i in range(2):
            with tru_app as rec:
                app.run(f"input_{i}")
                record_ids.extend([r.record_id for r in rec.records])

        session = TruSession()
        session.force_flush()
        session.wait_for_records(record_ids, timeout=10)

        # Emit decision spans manually: first record "evaluated",
        # second record "not_sampled".
        events_df = session.get_events(
            app_name="projection_test",
            app_version="v1",
            record_ids=record_ids,
            start_time=None,
        )

        _emit_sampling_decision_span(
            record_id=record_ids[0],
            app_name="projection_test",
            app_version="v1",
            events=events_df,
            sampling_meta={
                "sample_rate": 0.5,
                "eval_decision_reason": "evaluated",
                "sampled": True,
            },
        )
        _emit_sampling_decision_span(
            record_id=record_ids[1],
            app_name="projection_test",
            app_version="v1",
            events=events_df,
            sampling_meta={
                "sample_rate": 0.5,
                "eval_decision_reason": "not_sampled",
                "sampled": False,
            },
        )

        session.force_flush()
        # Give spans time to export.
        time.sleep(1)

        df, _ = session.get_records_and_feedback(
            app_name="projection_test",
            record_ids=record_ids,
        )

        self.assertEqual(len(df), 2)
        # Verify the sampled column exists and has correct values.
        self.assertIn("sampled", df.columns)
        self.assertIn("sample_rate", df.columns)
        self.assertIn("eval_decision_reason", df.columns)

        # Get rows by record_id.
        row_0 = df[df["record_id"] == record_ids[0]].iloc[0]
        row_1 = df[df["record_id"] == record_ids[1]].iloc[0]

        self.assertTrue(row_0["sampled"])
        self.assertEqual(row_0["sample_rate"], 0.5)
        self.assertEqual(row_0["eval_decision_reason"], "evaluated")

        self.assertFalse(row_1["sampled"])
        self.assertEqual(row_1["sample_rate"], 0.5)
        self.assertEqual(row_1["eval_decision_reason"], "not_sampled")

    def test_no_sampling_configured_returns_none(self):
        """Records without EVAL_DECISION spans should have sampled=None."""

        class _App:
            @instrument()
            def run(self, x: str) -> str:
                return f"echo {x}"

        app = _App()
        tru_app = TruApp(
            app,
            app_name="no_sampling_test",
            app_version="v1",
        )

        record_ids: List[str] = []
        with tru_app as rec:
            app.run("hello")
            record_ids.extend([r.record_id for r in rec.records])

        session = TruSession()
        session.force_flush()
        session.wait_for_records(record_ids, timeout=10)

        df, _ = session.get_records_and_feedback(
            app_name="no_sampling_test",
            record_ids=record_ids,
        )

        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0]["sampled"])
        self.assertIsNone(df.iloc[0]["sample_rate"])


@pytest.mark.optional
class TestBatchPathDoesNotChargeBudget(OtelTestCase):
    """A batch backfill must not burn the daily cost budget."""

    def _make_evaluator(self):
        mock_app = MagicMock()
        mock_app.app_name = "test_app"
        mock_app.app_version = "v1"
        mock_app.app_id = "app_hash_test"
        mock_app.connector = MagicMock()
        return Evaluator(mock_app), mock_app

    def test_compute_now_does_not_charge_budget(self):
        evaluator, mock_app = self._make_evaluator()

        session = TruSession()
        session.configure_online_eval(sample_rate=1.0, cost_budget=0.01)

        events = _make_mock_events()
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"test_record": events}
        )

        # Run via batch path (compute_now).
        evaluator._compute_feedbacks(in_evaluator_thread=False)

        # The budget should still be at 0 — batch didn't charge it.
        self.assertEqual(session.sampling_controller._daily_cost, 0.0)

        # Now verify that an ingest-path record still evaluates.
        evaluator._get_record_id_to_unprocessed_events = MagicMock(
            return_value={"ingest_record": events}
        )
        evaluator._compute_feedbacks(in_evaluator_thread=True)
        # Should have been called twice total (once batch, once ingest).
        self.assertEqual(mock_app.compute_feedbacks.call_count, 2)


if __name__ == "__main__":
    pytest.main([__file__])
