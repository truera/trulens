"""Regression test: a re-evaluated metric score must be stable and deterministic.

When a metric is re-evaluated on the same record there are several EVAL_ROOT
spans for one (record, metric). The records table set the score to whichever
span was iterated last, so the reported score changed with event order (and
diverged from the detailed result). The score must be deterministic: the latest
EVAL_ROOT by timestamp.
"""

from trulens.core.schema.event import Event
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

_APP = {
    "ai.observability.app_name": "app",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-1",
}


def _record_root(record_id, ts):
    return Event.model_validate({
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
            "ai.observability.span_type": SpanAttributes.SpanType.RECORD_ROOT.value,
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


def _eval_root(record_id, metric, score, ts, event_id):
    return Event.model_validate({
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
            "ai.observability.span_type": SpanAttributes.SpanType.EVAL_ROOT.value,
            SpanAttributes.EVAL_ROOT.METRIC_NAME: metric,
            SpanAttributes.EVAL_ROOT.SCORE: score,
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


class TestReevaluationScore(OtelTestCase):
    def _db_with_two_evals(self, order):
        from trulens.core.session import TruSession

        db = TruSession().connector.db
        db.reset_database()
        events = {
            "root": _record_root("r1", "2026-01-01T00:00:00"),
            "old": _eval_root(
                "r1", "Context Relevance", 0.67, "2026-01-01T00:00:05", "old"
            ),
            "new": _eval_root(
                "r1", "Context Relevance", 1.0, "2026-01-02T00:00:05", "new"
            ),
        }
        for key in order:
            db.insert_event(events[key])
        df, _ = db.get_records_and_feedback()
        return df.iloc[0]["Context Relevance"]

    def test_score_latest_when_old_first(self):
        # Older EVAL_ROOT inserted first: the latest (1.0) must still win.
        self.assertEqual(self._db_with_two_evals(["root", "old", "new"]), 1.0)

    def test_score_latest_when_new_first(self):
        # Newer EVAL_ROOT inserted first: the latest (1.0) must still win.
        self.assertEqual(self._db_with_two_evals(["root", "new", "old"]), 1.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
