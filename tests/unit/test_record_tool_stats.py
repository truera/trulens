"""Tests for deterministic per-record tool-use statistics."""

from datetime import datetime
from datetime import timedelta
import json

from trulens.core import session as core_session
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

APP = {
    "ai.observability.app_name": "tool-stats",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-tool-stats",
}


def _span(record_id, index, span_type, attributes):
    start = datetime(2026, 1, 1) + timedelta(seconds=index)
    return event_schema.Event.model_validate({
        "event_id": f"event-{record_id}-{index}",
        "record": {
            "kind": 1,
            "name": f"span-{index}",
            "parent_span_id": "",
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **APP,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.SPAN_TYPE: span_type.value,
            **attributes,
        },
        "record_type": "SPAN",
        "resource_attributes": APP,
        "start_timestamp": start,
        "timestamp": start + timedelta(milliseconds=100),
        "trace": {
            "parent_id": "",
            "span_id": f"span-{record_id}-{index}",
            "trace_id": f"trace-{record_id}",
        },
    })


def _root(record_id):
    return _span(
        record_id,
        0,
        SpanAttributes.SpanType.RECORD_ROOT,
        {
            SpanAttributes.RECORD_ROOT.INPUT: "input",
            SpanAttributes.RECORD_ROOT.OUTPUT: "output",
        },
    )


def _tool(record_id, index, name, arguments, *, error=False, mcp=False):
    span_type = (
        SpanAttributes.SpanType.MCP if mcp else SpanAttributes.SpanType.TOOL
    )
    if mcp:
        attributes = {
            SpanAttributes.MCP.TOOL_NAME: name,
            SpanAttributes.MCP.OUTPUT_IS_ERROR: error,
        }
        if arguments is not None:
            attributes[SpanAttributes.MCP.INPUT_ARGUMENTS] = arguments
    else:
        attributes = {GenAIAttributes.TOOL.NAME: name}
        if arguments is not None:
            attributes[GenAIAttributes.TOOL.CALL_ARGUMENTS] = arguments
        if error:
            attributes["error.type"] = "ToolError"
    return _span(record_id, index, span_type, attributes)


class TestRecordToolStats(OtelTestCase):
    def _records(self, events):
        db = core_session.TruSession().connector.db
        self.db = db
        for event in events:
            db.insert_event(event)
        records, _ = db.get_records_and_feedback()
        return records.set_index("record_id")

    def test_counts_repeats_and_immediate_retries(self):
        record_id = "record-with-tools"
        events = [
            _root(record_id),
            _tool(record_id, 1, "search", '{"q":"one"}'),
            _tool(record_id, 2, "fetch", '{"id":1}', error=True, mcp=True),
            _tool(record_id, 3, "fetch", '{"id":2}', mcp=True),
            _tool(record_id, 4, "save", '{"value":1}'),
            _tool(record_id, 5, "save", '{"value":1}'),
            _span(
                record_id,
                6,
                SpanAttributes.SpanType.GENERATION,
                {},
            ),
        ]

        row = self._records(events).loc[record_id]

        self.assertEqual(row["llm_calls"], 1)
        self.assertEqual(row["tool_calls"], 5)
        self.assertEqual(row["tool_errors"], 1)
        self.assertEqual(row["failed_then_retried"], 1)
        self.assertEqual(row["repeated_tool_calls"], 1)

        leaderboard, feedback_columns = self.db.get_leaderboard_aggregates()
        self.assertEqual(feedback_columns, [])
        self.assertEqual(
            leaderboard["Average Tool Calls per Record"].iloc[0], 5
        )
        self.assertEqual(leaderboard["Average Repeated Tool Calls"].iloc[0], 1)
        self.assertAlmostEqual(leaderboard["Tool Error Rate"].iloc[0], 0.2)

        trends = self.db.get_app_metric_trends()
        self.assertEqual(trends["tool_calls"].iloc[0], 5)
        self.assertEqual(trends["tool_errors"].iloc[0], 1)
        self.assertAlmostEqual(trends["tool_error_rate"].iloc[0], 0.2)

        public_leaderboard = core_session.TruSession().get_leaderboard(
            format="dict"
        )[0]
        self.assertEqual(public_leaderboard["average_tool_calls"], 5)
        self.assertEqual(public_leaderboard["average_repeated_tool_calls"], 1)
        self.assertAlmostEqual(public_leaderboard["tool_error_rate"], 0.2)
        json_leaderboard = json.loads(
            core_session.TruSession().get_leaderboard(format="json")
        )[0]
        self.assertEqual(json_leaderboard["average_tool_calls"], 5.0)
        self.assertAlmostEqual(json_leaderboard["tool_error_rate"], 0.2)

    def test_missing_arguments_makes_repeated_count_unknown(self):
        record_id = "record-without-payloads"
        row = self._records([
            _root(record_id),
            _tool(record_id, 1, "search", None),
            _tool(record_id, 2, "search", None),
        ]).loc[record_id]

        self.assertEqual(row["tool_calls"], 2)
        self.assertIsNone(row["repeated_tool_calls"])
