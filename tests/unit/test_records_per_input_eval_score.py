"""Regression test: a metric evaluated on several inputs of one record.

When a metric's selector matches several spans in one record (for example
context relevance over two retrieval spans), the feedback computer writes one
EVAL_ROOT span per input. The records view must report the mean of the latest
score of each input, not only the score of the latest EVAL_ROOT overall, while
re-evaluating the same input must still keep only its latest score.
"""

from trulens.core import session as core_session
from trulens.core.schema import event as event_schema
from trulens.otel.semconv import trace as trace_semconv

from tests.util import otel_test_case

_SpanAttributes = trace_semconv.SpanAttributes

_METRIC = "Context Relevance"

_APP = {
    "ai.observability.app_name": "app",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-1",
}


def _record_root(record_id, ts):
    return event_schema.Event.model_validate({
        "event_id": f"root-{record_id}",
        "record": {
            "kind": 1,
            "name": "q",
            "parent_span_id": "",
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **_APP,
            "ai.observability.record_id": record_id,
            "ai.observability.span_type": _SpanAttributes.SpanType.RECORD_ROOT.value,
            "ai.observability.record_root.input": "in",
            "ai.observability.record_root.output": "out",
        },
        "record_type": "SPAN",
        "resource_attributes": {"service.name": "trulens", **_APP},
        "start_timestamp": ts,
        "timestamp": ts,
        "trace": {
            "parent_id": "",
            "span_id": f"sr-{record_id}",
            "trace_id": f"tr-{record_id}",
        },
    })


def _eval_root(record_id, score, ts, event_id, context_span_id):
    """EVAL_ROOT for one input, identified by the span that supplied it."""
    return event_schema.Event.model_validate({
        "event_id": event_id,
        "record": {
            "kind": 1,
            "name": "eval",
            "parent_span_id": "",
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **_APP,
            "ai.observability.record_id": record_id,
            "ai.observability.span_type": _SpanAttributes.SpanType.EVAL_ROOT.value,
            _SpanAttributes.EVAL_ROOT.METRIC_NAME: _METRIC,
            _SpanAttributes.EVAL_ROOT.SCORE: score,
            f"{_SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID}.context": context_span_id,
            f"{_SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE}.context": (
                _SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS
            ),
        },
        "record_type": "SPAN",
        "resource_attributes": {"service.name": "trulens", **_APP},
        "start_timestamp": ts,
        "timestamp": ts,
        "trace": {
            "parent_id": "",
            "span_id": f"s-{event_id}",
            "trace_id": f"t-{event_id}",
        },
    })


class TestPerInputEvalScore(otel_test_case.OtelTestCase):
    def _score(self, events):
        db = core_session.TruSession().connector.db
        db.reset_database()
        db.insert_event(_record_root("r1", "2026-01-01T00:00:00"))
        for event in events:
            db.insert_event(event)
        df, feedback_cols = db.get_records_and_feedback()
        self.assertIn(_METRIC, feedback_cols)
        self.assertEqual(len(df), 1)
        return df.iloc[0][_METRIC]

    def test_two_inputs_are_averaged(self):
        score = self._score([
            _eval_root("r1", 1.0, "2026-01-01T00:00:05", "a", "ret-a"),
            _eval_root("r1", 0.0, "2026-01-01T00:00:06", "b", "ret-b"),
        ])
        self.assertAlmostEqual(score, 0.5)

    def test_two_inputs_are_averaged_in_any_order(self):
        score = self._score([
            _eval_root("r1", 0.0, "2026-01-01T00:00:06", "b", "ret-b"),
            _eval_root("r1", 1.0, "2026-01-01T00:00:05", "a", "ret-a"),
        ])
        self.assertAlmostEqual(score, 0.5)

    def test_same_input_reevaluated_keeps_latest(self):
        for order in (("old", "new"), ("new", "old")):
            with self.subTest(order=order):
                events = {
                    "old": _eval_root(
                        "r1", 0.2, "2026-01-01T00:00:05", "old", "ret-a"
                    ),
                    "new": _eval_root(
                        "r1", 0.8, "2026-01-02T00:00:05", "new", "ret-a"
                    ),
                }
                score = self._score([events[key] for key in order])
                self.assertAlmostEqual(score, 0.8)

    def test_reevaluated_input_mixed_with_other_input(self):
        # Input A is evaluated twice (0.2 then 0.8), input B once (0.0). The
        # score is mean(latest A, B) = 0.4, regardless of insertion order.
        events = {
            "a_old": _eval_root(
                "r1", 0.2, "2026-01-01T00:00:05", "a_old", "ret-a"
            ),
            "b": _eval_root("r1", 0.0, "2026-01-01T00:00:06", "b", "ret-b"),
            "a_new": _eval_root(
                "r1", 0.8, "2026-01-02T00:00:05", "a_new", "ret-a"
            ),
        }
        for order in (
            ("a_old", "b", "a_new"),
            ("a_new", "b", "a_old"),
            ("b", "a_new", "a_old"),
        ):
            with self.subTest(order=order):
                score = self._score([events[key] for key in order])
                self.assertAlmostEqual(score, 0.4)

    def test_missing_score_is_excluded_from_mean(self):
        score = self._score([
            _eval_root("r1", 0.6, "2026-01-01T00:00:05", "a", "ret-a"),
            _eval_root("r1", None, "2026-01-01T00:00:06", "b", "ret-b"),
        ])
        self.assertAlmostEqual(score, 0.6)


if __name__ == "__main__":
    import unittest

    unittest.main()
