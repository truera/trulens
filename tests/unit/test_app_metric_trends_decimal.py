"""Regression test: app metric trends accept Decimal latencies.

On Postgres the latency expression is `EXTRACT(EPOCH FROM ...)`, which
returns `numeric`, so the driver hands back `decimal.Decimal` instead of
float. The p90 and p99 quantiles then raised `TypeError` because numpy cannot
multiply a Decimal by a float. SQLite returns floats, so the SQLite test suite
never saw it. Here the SQLite latency expression is coerced to `sa.Numeric`
so its results come back as Decimal, the same as on Postgres.
"""

import decimal
from unittest import mock
import warnings

import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from trulens.core import session as core_session
from trulens.core.schema import event as event_schema
from trulens.otel.semconv import trace as semconv_trace

from tests.util.otel_test_case import OtelTestCase

SpanAttributes = semconv_trace.SpanAttributes

APP = {
    "ai.observability.app_name": "decimal_latency",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-decimal-latency",
}


def _root(record_id, seconds):
    return event_schema.Event.model_validate({
        "event_id": f"evt-{record_id}",
        "record": {
            "kind": 1,
            "name": "root",
            "parent_span_id": "",
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **APP,
            "ai.observability.record_id": record_id,
            "ai.observability.span_type": (
                SpanAttributes.SpanType.RECORD_ROOT.value
            ),
            "ai.observability.record_root.input": "in",
            "ai.observability.record_root.output": "out",
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
        "timestamp": f"2026-01-01T00:00:{seconds:02d}",
        "trace": {
            "parent_id": "",
            "span_id": f"span-{record_id}",
            "trace_id": f"trace-{record_id}",
        },
    })


class TestAppMetricTrendsDecimalLatency(OtelTestCase):
    def test_decimal_latency_is_aggregated(self):
        db = core_session.TruSession().connector.db
        for i, seconds in enumerate([1, 2, 3, 4]):
            db.insert_event(_root(f"r{i}", seconds))

        latency_expr = db._latency_seconds_expr

        def decimal_latency_expr():
            return sa.type_coerce(latency_expr(), sa.Numeric(asdecimal=True))

        with (
            mock.patch.object(
                db, "_latency_seconds_expr", decimal_latency_expr
            ),
            warnings.catch_warnings(),
        ):
            # SQLite warns that it has no native Decimal; that is the point.
            warnings.simplefilter("ignore", sa_exc.SAWarning)
            with db.session.begin() as session:
                raw = session.execute(
                    sa.select(db._latency_seconds_expr())
                ).scalar()
            self.assertIsInstance(raw, decimal.Decimal)
            trends = db.get_app_metric_trends()

        self.assertEqual(trends["record_count"].tolist(), [4])
        self.assertAlmostEqual(trends["average_latency"].iloc[0], 2.5, places=3)
        self.assertAlmostEqual(trends["p90_latency"].iloc[0], 3.7, places=3)
        self.assertAlmostEqual(trends["p99_latency"].iloc[0], 3.97, places=3)
        self.assertIsInstance(trends["p90_latency"].iloc[0], float)


if __name__ == "__main__":
    import unittest

    unittest.main()
