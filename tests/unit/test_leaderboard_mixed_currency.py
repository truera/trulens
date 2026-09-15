"""Regression test: the leaderboard must not sum costs across currencies.

The OTel leaderboard summed every record's cost into one number and labelled it
with whichever currency sorted last (max), so an app with records in USD and in
Snowflake credits had both amounts added together and attributed to a single
currency. Cost per currency must be reported separately.
"""

from trulens.core.schema.event import Event
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestLeaderboardMixedCurrency(OtelTestCase):
    def _record_root(self, record_id, cost, currency):
        app = {
            "ai.observability.app_name": "mixed",
            "ai.observability.app_version": "v1",
            "ai.observability.app_id": "app-mixed",
        }
        return Event.model_validate({
            "event_id": f"evt-{record_id}",
            "record": {
                "kind": 1,
                "name": "app.query",
                "parent_span_id": "",
                "status": "STATUS_CODE_UNSET",
            },
            "record_attributes": {
                **app,
                "ai.observability.record_id": record_id,
                "ai.observability.span_type": SpanAttributes.SpanType.RECORD_ROOT.value,
                "ai.observability.record_root.input": "in",
                "ai.observability.record_root.output": "out",
                SpanAttributes.COST.COST: cost,
                SpanAttributes.COST.CURRENCY: currency,
                SpanAttributes.COST.NUM_TOKENS: 10,
            },
            "record_type": "SPAN",
            "resource_attributes": {
                "service.name": "trulens",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.version": "1.31.0",
                **app,
            },
            "start_timestamp": "2026-01-01T00:00:00",
            "timestamp": "2026-01-01T00:00:01",
            "trace": {
                "parent_id": "",
                "span_id": f"span-{record_id}",
                "trace_id": f"trace-{record_id}",
            },
        })

    def test_costs_are_reported_per_currency(self):
        from trulens.core.session import TruSession

        db = TruSession().connector.db
        db.insert_event(self._record_root("r1", 0.03, "USD"))
        db.insert_event(self._record_root("r2", 0.05, "Snowflake credits"))

        leaderboard, _ = db.get_leaderboard_aggregates()
        self.assertEqual(len(leaderboard), 1)
        row = leaderboard.iloc[0]

        # USD and Snowflake-credit costs must not be summed together.
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.03)
        self.assertAlmostEqual(row["Total Cost (Snowflake Credits)"], 0.05)


if __name__ == "__main__":
    import unittest

    unittest.main()
