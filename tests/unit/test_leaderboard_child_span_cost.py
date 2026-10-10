"""Regression test: OTEL leaderboard and app metric trends count child cost.

Providers write cost on the GENERATION span and client hooks on the AGENT
span, not on the RECORD_ROOT. The leaderboard and the app metric trends only
read cost from RECORD_ROOT spans, so they showed zero cost and tokens for
instrumented apps while the records view summed cost over the record's spans.
"""

from trulens.core import session as core_session
from trulens.core.schema import event as event_schema
from trulens.otel.semconv import trace as semconv_trace

from tests.util.otel_test_case import OtelTestCase

SpanAttributes = semconv_trace.SpanAttributes
SpanType = SpanAttributes.SpanType

APP = {
    "ai.observability.app_name": "child_cost",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-child-cost",
}


def _cost(cost, tokens, currency="USD"):
    return {
        SpanAttributes.COST.COST: cost,
        SpanAttributes.COST.CURRENCY: currency,
        SpanAttributes.COST.NUM_TOKENS: tokens,
    }


def _span(record_id, name, span_type, attributes=None, parent=""):
    return event_schema.Event.model_validate({
        "event_id": f"evt-{record_id}-{name}",
        "record": {
            "kind": 1,
            "name": name,
            "parent_span_id": parent,
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **APP,
            "ai.observability.record_id": record_id,
            "ai.observability.span_type": span_type.value,
            **(attributes or {}),
        },
        "record_type": "SPAN",
        "resource_attributes": {
            "service.name": "trulens",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.31.0",
            **APP,
        },
        "start_timestamp": "2026-01-01T00:00:00",
        "timestamp": "2026-01-01T00:00:01",
        "trace": {
            "parent_id": parent,
            "span_id": f"span-{record_id}-{name}",
            "trace_id": f"trace-{record_id}",
        },
    })


def _root(record_id, attributes=None):
    return _span(
        record_id,
        "root",
        SpanType.RECORD_ROOT,
        {
            "ai.observability.record_root.input": "in",
            "ai.observability.record_root.output": "out",
            **(attributes or {}),
        },
    )


def _child(record_id, name, span_type, attributes):
    return _span(
        record_id,
        name,
        span_type,
        attributes,
        parent=f"span-{record_id}-root",
    )


class TestLeaderboardChildSpanCost(OtelTestCase):
    def _db(self, events):
        db = core_session.TruSession().connector.db
        for event in events:
            db.insert_event(event)
        return db

    def _leaderboard_row(self, db):
        leaderboard, _ = db.get_leaderboard_aggregates()
        self.assertEqual(len(leaderboard), 1)
        return leaderboard.iloc[0]

    def test_generation_child_cost_is_counted(self):
        events = []
        for i in range(3):
            events.append(_root(f"r{i}"))
            events.append(
                _child(f"r{i}", "llm", SpanType.GENERATION, _cost(0.02, 100))
            )
        db = self._db(events)

        records, _ = db.get_records_and_feedback()
        self.assertAlmostEqual(records["total_cost"].sum(), 0.06)
        self.assertEqual(records["total_tokens"].sum(), 300)

        row = self._leaderboard_row(db)
        self.assertEqual(row["Records"], 3)
        self.assertAlmostEqual(
            row["Total Cost (USD)"], records["total_cost"].sum()
        )
        self.assertAlmostEqual(
            row["Total Tokens"], records["total_tokens"].sum()
        )
        self.assertAlmostEqual(row["Total Cost (Snowflake Credits)"], 0.0)

        trends = db.get_app_metric_trends()
        self.assertEqual(trends["currency"].tolist(), ["USD"])
        self.assertEqual(trends["record_count"].tolist(), [3])
        self.assertAlmostEqual(trends["total_app_cost"].iloc[0], 0.06)
        self.assertAlmostEqual(trends["average_app_cost"].iloc[0], 0.02)

    def test_eval_span_cost_is_not_counted(self):
        eval_root = {
            SpanAttributes.EVAL_ROOT.METRIC_NAME: "relevance",
            SpanAttributes.EVAL_ROOT.SCORE: 1.0,
            **_cost(0.5, 1000),
        }
        db = self._db([
            _root("r1"),
            _child("r1", "llm", SpanType.GENERATION, _cost(0.02, 100)),
            _child("r1", "eval_root", SpanType.EVAL_ROOT, eval_root),
            _child("r1", "eval", SpanType.EVAL, _cost(0.25, 500)),
        ])

        records, _ = db.get_records_and_feedback()
        self.assertAlmostEqual(records["total_cost"].sum(), 0.02)
        self.assertEqual(records["total_tokens"].sum(), 100)

        row = self._leaderboard_row(db)
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.02)
        self.assertAlmostEqual(row["Total Tokens"], 100)

        trends = db.get_app_metric_trends()
        self.assertEqual(trends["record_count"].tolist(), [1])
        self.assertAlmostEqual(trends["total_app_cost"].iloc[0], 0.02)

    def test_root_cost_without_children(self):
        db = self._db([_root("r1", _cost(0.03, 10))])

        row = self._leaderboard_row(db)
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.03)
        self.assertAlmostEqual(row["Total Tokens"], 10)

        trends = db.get_app_metric_trends()
        self.assertAlmostEqual(trends["total_app_cost"].iloc[0], 0.03)

    def test_root_and_child_cost_are_each_counted_once(self):
        db = self._db([
            _root("r1", _cost(0.01, 5)),
            _child("r1", "agent", SpanType.AGENT, _cost(0.02, 100)),
        ])

        records, _ = db.get_records_and_feedback()
        row = self._leaderboard_row(db)
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.03)
        self.assertAlmostEqual(
            row["Total Cost (USD)"], records["total_cost"].sum()
        )
        self.assertAlmostEqual(row["Total Tokens"], 105)

        trends = db.get_app_metric_trends()
        self.assertEqual(trends["record_count"].tolist(), [1])
        self.assertAlmostEqual(trends["total_app_cost"].iloc[0], 0.03)

    def test_child_cost_keeps_its_currency(self):
        db = self._db([
            _root("r1"),
            _child(
                "r1",
                "llm",
                SpanType.GENERATION,
                _cost(0.05, 10, "Snowflake credits"),
            ),
        ])

        row = self._leaderboard_row(db)
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.0)
        self.assertAlmostEqual(row["Total Cost (Snowflake Credits)"], 0.05)

        trends = db.get_app_metric_trends()
        self.assertEqual(trends["currency"].tolist(), ["Snowflake credits"])
        self.assertEqual(trends["record_count"].tolist(), [1])
        self.assertAlmostEqual(trends["total_app_cost"].iloc[0], 0.05)


if __name__ == "__main__":
    import unittest

    unittest.main()
