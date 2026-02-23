"""
Snowflake integration tests for metrics usage fixes (discussion #2060):

1. UserWarning (not ValueError) when metrics passed to TruApp with
   Snowflake account-level event table + OTEL.
2. _should_skip_computation correctly handles Metric objects (not just
   strings) via .name extraction.
"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

try:
    from trulens.core.run import Run
except Exception:
    Run = None

try:
    from trulens.core.feedback.selector import Selector
    from trulens.core.metric.metric import Metric
    from trulens.otel.semconv.trace import SpanAttributes
except Exception:
    Metric = None
    Selector = None
    SpanAttributes = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(**overrides) -> "Run":
    """Build a minimal Run instance for unit testing."""
    base = {
        "run_name": "test_run",
        "object_name": "TEST_AGENT",
        "object_type": "EXTERNAL AGENT",
        "object_version": "v1",
        "run_metadata": {},
        "source_info": {
            "name": "dummy_source",
            "column_spec": {"input": "INPUT"},
            "source_type": "TABLE",
        },
        "app": MagicMock(),
        "main_method_name": "dummy_method",
        "run_dao": MagicMock(),
        "tru_session": MagicMock(),
    }
    base.update(overrides)
    return Run.model_validate(base)


def _make_metric(name: str = "my_metric") -> "Metric":
    """Build a minimal Metric instance for unit testing."""

    def impl(output: str) -> float:
        return 1.0

    return Metric(
        implementation=impl,
        name=name,
        selectors={
            "output": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
            )
        },
    )


# ---------------------------------------------------------------------------
# Bug 1: warning instead of raise in App._tru_post_init
# ---------------------------------------------------------------------------


@pytest.mark.snowflake
class TestTruPostInitSnowflakeOtelWarning(unittest.TestCase):
    """
    Verify that App._tru_post_init emits a UserWarning (not ValueError)
    when feedbacks are present and the Snowflake account-level event table
    + OTEL path is active.

    We call _tru_post_init directly on a mock self to avoid the full TruApp
    constructor chain.
    """

    def _call_tru_post_init(self, feedbacks):
        """Call App._tru_post_init with Snowflake+OTEL path active."""
        from trulens.core.app import App
        from trulens.core.schema import feedback as feedback_schema

        mock_self = MagicMock()
        mock_self.connector = MagicMock()
        mock_self.feedbacks = feedbacks
        mock_self.feedback_mode = feedback_schema.FeedbackMode.WITH_APP
        mock_self.selector_nocheck = True
        # Trigger the Snowflake+OTEL else-branch
        mock_self._is_account_level_event_table_snowflake_connector.return_value = True

        with patch(
            "trulens.core.app.is_otel_tracing_enabled", return_value=True
        ):
            App._tru_post_init(mock_self)

    def test_no_warning_without_feedbacks(self):
        """No warning emitted when feedbacks list is empty."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_tru_post_init([])

        user_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertEqual(len(user_warnings), 0)

    def test_warning_emitted_with_feedbacks(self):
        """UserWarning is emitted when metrics are present in Snowflake+OTEL mode."""
        if Metric is None or Selector is None:
            self.skipTest("Metric/Selector not available.")

        metric = _make_metric()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_tru_post_init([metric])

        user_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertGreater(
            len(user_warnings),
            0,
            "Expected a UserWarning when metrics are passed in Snowflake+OTEL mode.",
        )
        warning_text = str(user_warnings[0].message)
        self.assertIn("Metrics passed to TruApp", warning_text)
        self.assertIn("compute_metrics", warning_text)

    def test_no_value_error_raised(self):
        """_tru_post_init must not raise ValueError when feedbacks are present."""
        if Metric is None or Selector is None:
            self.skipTest("Metric/Selector not available.")

        metric = _make_metric()

        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self._call_tru_post_init([metric])
        except ValueError as exc:
            self.fail(f"_tru_post_init raised ValueError unexpectedly: {exc}")

    def test_check_otel_selectors_called(self):
        """check_otel_selectors() is invoked on each feedback after the warning."""
        if Metric is None or Selector is None:
            self.skipTest("Metric/Selector not available.")

        mock_metric = MagicMock()
        mock_metric.run_location = None

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._call_tru_post_init([mock_metric])

        mock_metric.check_otel_selectors.assert_called_once()


# ---------------------------------------------------------------------------
# Bug 2: _should_skip_computation handles Metric objects
# ---------------------------------------------------------------------------


@pytest.mark.snowflake
class TestShouldSkipComputationWithMetricObjects(unittest.TestCase):
    """
    Verify that _should_skip_computation extracts .name from Metric objects
    so that deduplication works correctly regardless of whether a string or
    a Metric instance is passed.
    """

    def setUp(self):
        if Run is None:
            self.skipTest("Run not available.")
        if Metric is None or Selector is None:
            self.skipTest("Metric/Selector not available.")

        self.run = _make_run()
        self.mock_metrics = {
            "met1": {
                "name": "my_metric",
                "completion_status": {
                    "status": Run.CompletionStatusStatus.COMPLETED,
                    "record_count": 10,
                },
            }
        }

    def _patch_describe(self):
        return patch.object(
            Run,
            "describe",
            return_value={"run_metadata": {"metrics": self.mock_metrics}},
        )

    # -- passing a string (existing behaviour, must still work) --

    def test_string_completed_skips(self):
        with self._patch_describe():
            result = self.run._should_skip_computation("my_metric", self.run)
        self.assertTrue(result)

    def test_string_no_match_does_not_skip(self):
        with self._patch_describe():
            result = self.run._should_skip_computation("other_metric", self.run)
        self.assertFalse(result)

    # -- passing a Metric object (new behaviour) --

    def test_metric_object_completed_skips(self):
        """A completed Metric object should be skipped."""
        metric = _make_metric(name="my_metric")
        with self._patch_describe():
            result = self.run._should_skip_computation(metric, self.run)
        self.assertTrue(result)

    def test_metric_object_no_match_does_not_skip(self):
        """A Metric whose name is not in run metadata should not be skipped."""
        metric = _make_metric(name="other_metric")
        with self._patch_describe():
            result = self.run._should_skip_computation(metric, self.run)
        self.assertFalse(result)

    def test_metric_object_failed_does_not_skip(self):
        """A Metric where all entries are FAILED should allow re-computation."""
        self.mock_metrics["met1"]["completion_status"]["status"] = (
            Run.CompletionStatusStatus.FAILED
        )
        metric = _make_metric(name="my_metric")
        with self._patch_describe():
            result = self.run._should_skip_computation(metric, self.run)
        self.assertFalse(result)

    def test_metric_object_in_progress_skips(self):
        """A Metric where an entry is STARTED should be skipped."""
        self.mock_metrics["met1"]["completion_status"]["status"] = (
            Run.CompletionStatusStatus.STARTED
        )
        metric = _make_metric(name="my_metric")
        with self._patch_describe():
            result = self.run._should_skip_computation(metric, self.run)
        self.assertTrue(result)
