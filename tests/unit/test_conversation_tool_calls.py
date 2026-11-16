"""Tests for tool calls selected as part of conversation turns."""

from datetime import datetime
from datetime import timedelta

import pandas as pd
from trulens.feedback import computer
from trulens.otel.semconv.trace import ErrorAttributes
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes


def _event(
    *,
    span_id: str,
    record_id: str,
    start_timestamp: datetime,
    span_type: str,
    attributes: dict | None = None,
) -> dict:
    return {
        "trace": {"span_id": span_id},
        "start_timestamp": start_timestamp,
        "record_attributes": {
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.SPAN_TYPE: span_type,
            **(attributes or {}),
        },
    }


def test_tool_spans_are_ordered_and_use_standard_attributes() -> None:
    start = datetime.now()
    events = pd.DataFrame([
        _event(
            span_id="later",
            record_id="turn-2",
            start_timestamp=start + timedelta(seconds=2),
            span_type=SpanAttributes.SpanType.TOOL,
            attributes={
                GenAIAttributes.TOOL.NAME: "test",
                GenAIAttributes.TOOL.CALL_ARGUMENTS: '{"suite": "unit"}',
                GenAIAttributes.TOOL.CALL_RESULT: "failed",
                ErrorAttributes.TYPE: "test_failure",
            },
        ),
        _event(
            span_id="earlier",
            record_id="turn-2",
            start_timestamp=start + timedelta(seconds=1),
            span_type=SpanAttributes.SpanType.TOOL,
            attributes={
                GenAIAttributes.TOOL.NAME: "edit",
                GenAIAttributes.TOOL.CALL_ARGUMENTS: '{"file": "app.py"}',
                GenAIAttributes.TOOL.CALL_RESULT: "updated",
            },
        ),
    ])

    spans = computer._conversation_tool_spans_by_record_id(events)["turn-2"]
    calls = [computer._tool_call_from_span(span, 2_000) for span in spans]

    assert [call["name"] for call in calls] == ["edit", "test"]
    assert calls[1]["error"] == "test_failure"


def test_mcp_fallback_and_tool_result_truncation() -> None:
    result = "x" * 50_000
    span = pd.Series(
        _event(
            span_id="mcp",
            record_id="turn-1",
            start_timestamp=datetime.now(),
            span_type=SpanAttributes.SpanType.MCP,
            attributes={
                SpanAttributes.MCP.TOOL_NAME: "search",
                SpanAttributes.MCP.INPUT_ARGUMENTS: '{"query": "TruLens"}',
                SpanAttributes.MCP.OUTPUT_CONTENT: result,
                SpanAttributes.MCP.OUTPUT_IS_ERROR: True,
            },
        )
    )

    call = computer._tool_call_from_span(span, 2_000)

    assert call == {
        "name": "search",
        "arguments": '{"query": "TruLens"}',
        "result": "x" * 2_000 + "... [truncated]",
        "error": "mcp_tool_error",
    }


def test_falsy_tool_error_is_normalized_to_none() -> None:
    span = pd.Series(
        _event(
            span_id="successful-tool",
            record_id="turn-1",
            start_timestamp=datetime.now(),
            span_type=SpanAttributes.SpanType.TOOL,
            attributes={
                GenAIAttributes.TOOL.NAME: "search",
                GenAIAttributes.TOOL.CALL_RESULT: "ok",
                SpanAttributes.CALL.ERROR: False,
            },
        )
    )

    call = computer._tool_call_from_span(span, 2_000)

    assert call["error"] is None
