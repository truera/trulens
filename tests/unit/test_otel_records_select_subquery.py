"""Regression test: paginated OTEL records query must not use `Select.c`.

`SQLAlchemyDB._get_records_and_feedback_otel` read the `record_id` column off
the paginated record-id SELECT via `.c`, which SQLAlchemy deprecated in 1.4
(it implicitly creates a subquery) and removed in 2.1. On SQLAlchemy 2.1 every
`get_records_and_feedback()` call without explicit record ids failed with
`AttributeError: 'Select' object has no attribute 'c'`.
"""

import warnings

from sqlalchemy.exc import SADeprecationWarning
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


class TestOtelRecordsSelectSubquery(OtelTestCase):
    def test_paginated_records_query_uses_explicit_subquery(self):
        from trulens.core.session import TruSession

        db = TruSession().connector.db
        db.reset_database()
        for i in range(3):
            db.insert_event(_record_root(f"r{i}", f"2026-01-01T00:00:0{i}"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", SADeprecationWarning)
            df, _ = db.get_records_and_feedback()
            paged, _ = db.get_records_and_feedback(offset=1, limit=1)

        self.assertEqual(sorted(df["record_id"]), ["r0", "r1", "r2"])
        self.assertEqual(len(paged), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
