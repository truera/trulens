"""Tests for Claude Code and Cursor hook instrumentation."""

from __future__ import annotations

from datetime import timedelta
from datetime import timezone
import json
from pathlib import Path

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest
from trulens.core.otel.client_hooks import journal
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import privacy
from trulens.core.otel.client_hooks import service
from trulens.core.otel.client_hooks import tracing
from trulens.experimental.otel_tracing.core.exporter import (
    utils as exporter_utils,
)
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import GenAIEvents
from trulens.otel.semconv.trace import SpanAttributes


def _claude(event_name: str, **values):
    return {
        "session_id": "session-1",
        "hook_event_name": event_name,
        **values,
    }


def _cursor(event_name: str, **values):
    return {
        "conversation_id": "conversation-1",
        "generation_id": "generation-1",
        "hook_event_name": event_name,
        **values,
    }


def test_claude_parser_normalizes_tool_failure():
    event = parsers.parse_claude(
        _claude(
            "PostToolUseFailure",
            tool_name="Bash",
            tool_use_id="tool-1",
            tool_input={"command": "pytest"},
            error="exit 1",
            duration_ms=10,
        )
    )

    assert event.category == "tool"
    assert event.phase == "end"
    assert event.failed
    assert event.operation_id == "tool-1"
    assert event.duration_ms == 10


def test_cursor_parser_preserves_unknown_metadata():
    event = parsers.parse_cursor(
        _cursor("customFutureEvent", future_field="future-value")
    )

    assert event.category == "workflow"
    assert event.metadata == {"future_field": "future-value"}


def test_capture_policy_is_metadata_only_by_default():
    event = parsers.parse_claude(
        _claude(
            "UserPromptSubmit",
            prompt="secret prompt",
            transcript_path="/secret/transcript.jsonl",
        )
    )

    captured = privacy.CapturePolicy().apply(event)

    assert captured.prompt is None
    assert captured.paths is None


def test_capture_policy_redacts_and_bounds_content():
    event = parsers.parse_claude(
        _claude(
            "PostToolUse",
            tool_name="MCP:example",
            tool_input={"token": "secret", "value": "x" * 500},
        )
    )

    captured = privacy.CapturePolicy(
        capture_tool_payloads=True, max_field_bytes=256
    ).apply(event)

    serialized = json.dumps(captured.tool_input)
    assert "secret" not in serialized
    assert "[REDACTED]" in serialized
    assert "[TRUNCATED]" in serialized


def test_capture_policy_hides_error_content_by_default():
    event = parsers.parse_claude(
        _claude("PostToolUseFailure", error="token=secret")
    )

    captured = privacy.CapturePolicy().apply(event)

    assert captured.error == "[error content not captured]"


def test_cursor_file_edit_captures_diff_only_when_enabled():
    event = parsers.parse_cursor(
        _cursor(
            "afterFileEdit",
            file_path="src/app.py",
            edits=[
                {
                    "old_string": "password = 'secret'",
                    "new_string": "password = os.environ['PASSWORD']",
                }
            ],
        )
    )

    assert event.category == "tool"
    assert privacy.CapturePolicy().apply(event).diff is None
    captured = privacy.CapturePolicy(
        capture_diffs=True, capture_paths=True
    ).apply(event)
    assert captured.diff["file_path"] == "src/app.py"
    assert captured.paths["file_path"] == "src/app.py"


def test_apply_patch_diff_is_bounded_and_emitted_on_span():
    event = parsers.parse_claude(
        _claude(
            "PostToolUse",
            tool_name="ApplyPatch",
            tool_use_id="patch-1",
            tool_input={"patch": "@@\n- old\n+ new\n" + "x" * 500},
        )
    )
    captured = privacy.CapturePolicy(
        capture_diffs=True, max_field_bytes=256
    ).apply(event)

    spans = tracing.TraceAssembler().assemble([captured])
    patch_span = next(span for span in spans if span.name == "ApplyPatch")
    assert (
        "[TRUNCATED]" in patch_span.attributes[SpanAttributes.CODING_AGENT.DIFF]
    )
    assert (
        patch_span.attributes["vcs.change.diff"]
        == patch_span.attributes[SpanAttributes.CODING_AGENT.DIFF]
    )


def test_journal_deduplicates_and_correlates_turn(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)
    prompt = privacy.CapturePolicy(capture_content=True).apply(
        parsers.parse_claude(_claude("UserPromptSubmit", prompt="hello"))
    )
    tool = parsers.parse_claude(
        _claude("PreToolUse", tool_name="Read", tool_use_id="tool-1")
    )
    stop = parsers.parse_claude(_claude("Stop"))

    turn_id, _ = event_journal.append(prompt)
    event_journal.append(tool)
    event_journal.append(tool)
    event_journal.append(stop)

    turn = event_journal.get_turn("claude", "session-1", turn_id)
    assert [event.event_name for event in turn] == [
        "UserPromptSubmit",
        "PreToolUse",
        "Stop",
    ]
    assert event_journal.pending_turns(
        "claude", "session-1", stale_after=timedelta(days=1)
    ) == [turn_id]


def test_journal_gives_identical_prompts_distinct_turn_ids(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)

    first, _ = event_journal.append(
        parsers.parse_claude(_claude("UserPromptSubmit", prompt="same"))
    )
    event_journal.append(parsers.parse_claude(_claude("Stop")))
    event_journal.mark_exported("claude", "session-1", first)
    second, _ = event_journal.append(
        parsers.parse_claude(_claude("UserPromptSubmit", prompt="same"))
    )

    assert second != first


def test_journal_deduplicates_retried_prompt_before_stop(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)
    prompt = parsers.parse_claude(_claude("UserPromptSubmit", prompt="same"))

    first, _ = event_journal.append(prompt)
    second, _ = event_journal.append(prompt)

    assert second == first
    assert len(event_journal.get_turn("claude", "session-1", first)) == 1


def test_journal_claims_pending_turn_once(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)
    turn_id, _ = event_journal.append(
        parsers.parse_claude(_claude("UserPromptSubmit"))
    )
    event_journal.append(parsers.parse_claude(_claude("Stop")))

    assert event_journal.claim_pending_turns("claude", "session-1") == [turn_id]
    assert event_journal.claim_pending_turns("claude", "session-1") == []


def test_journal_groups_sequential_turns_by_conversation(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)
    first_turn, _ = event_journal.append(
        parsers.parse_cursor(_cursor("beforeSubmitPrompt", prompt="first"))
    )
    event_journal.append(parsers.parse_cursor(_cursor("stop")))
    second_payload = _cursor("beforeSubmitPrompt", prompt="second")
    second_payload["generation_id"] = "generation-2"
    second_turn, _ = event_journal.append(parsers.parse_cursor(second_payload))

    first_events = event_journal.get_turn(
        "cursor", "conversation-1", first_turn
    )
    second_events = event_journal.get_turn(
        "cursor", "conversation-1", second_turn
    )
    assert first_events[0].conversation_id == second_events[0].conversation_id
    assert first_events[0].turn_id != second_events[0].turn_id


def test_assembler_creates_private_root_agent_and_tool_spans():
    policy = privacy.CapturePolicy()
    events = [
        policy.apply(
            parsers.parse_cursor(
                _cursor("beforeSubmitPrompt", prompt="private prompt")
            )
        ),
        policy.apply(
            parsers.parse_cursor(
                _cursor(
                    "beforeShellExecution",
                    command="pytest",
                    operation_id="shell-1",
                )
            )
        ),
        policy.apply(
            parsers.parse_cursor(
                _cursor(
                    "afterShellExecution",
                    output="private output",
                    duration=20,
                    operation_id="shell-1",
                )
            )
        ),
        policy.apply(parsers.parse_cursor(_cursor("stop", status="completed"))),
    ]

    spans = tracing.TraceAssembler().assemble(events)

    root = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.RECORD_ROOT.value
    )
    agent = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.AGENT.value
    )
    tool = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.TOOL.value
    )
    assert root.attributes[SpanAttributes.RECORD_ROOT.INPUT] == (
        "[content not captured]"
    )
    assert root.attributes[SpanAttributes.RUN_NAME] == "client-hooks"
    assert root.attributes[SpanAttributes.INPUT_RECORDS_COUNT] == 1
    assert (
        root.context.trace_id == agent.context.trace_id == tool.context.trace_id
    )
    assert root.parent is None
    assert agent.parent.span_id == root.context.span_id
    assert tool.parent.span_id == agent.context.span_id
    assert GenAIAttributes.TOOL.CALL_ARGUMENTS not in tool.attributes


def test_cursor_after_response_waits_for_stop_and_captures_text():
    policy = privacy.CapturePolicy(capture_content=True)
    events = [
        policy.apply(
            parsers.parse_cursor(_cursor("beforeSubmitPrompt", prompt="hello"))
        ),
        policy.apply(
            parsers.parse_cursor(
                _cursor("afterAgentResponse", text="final response")
            )
        ),
        policy.apply(parsers.parse_cursor(_cursor("stop", status="completed"))),
    ]

    assert not events[1].terminal
    spans = tracing.TraceAssembler().assemble(events)
    root = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.RECORD_ROOT.value
    )
    assert (
        root.attributes[SpanAttributes.RECORD_ROOT.OUTPUT] == "final response"
    )
    assert root.attributes[SpanAttributes.CALL.RETURN] == '"final response"'
    assert (
        "final response"
        in root.events[0].attributes[
            GenAIEvents.EventAttributes.OUTPUT_MESSAGES
        ]
    )
    assert json.loads(
        root.events[0].attributes[GenAIEvents.EventAttributes.INPUT_MESSAGES]
    ) == [{"role": "user", "content": "hello"}]
    assert json.loads(
        root.events[0].attributes[GenAIEvents.EventAttributes.OUTPUT_MESSAGES]
    ) == [{"role": "assistant", "content": "final response"}]
    response_span = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.GENERATION.value
    )
    agent = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.AGENT.value
    )
    assert response_span.parent.span_id == agent.context.span_id
    assert response_span.attributes[SpanAttributes.CALL.RETURN] == (
        '"final response"'
    )


def test_parser_normalizes_timezone_less_timestamp_to_utc():
    event = parsers.parse_claude(
        _claude("UserPromptSubmit", timestamp="2026-08-25T12:00:00")
    )

    assert event.observed_at.tzinfo == timezone.utc


def test_assembler_marks_stale_turn_as_error():
    events = [parsers.parse_claude(_claude("UserPromptSubmit"))]

    spans = tracing.TraceAssembler().assemble(events, stale=True)

    root = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.RECORD_ROOT.value
    )
    assert root.status.status_code.name == "ERROR"
    assert root.attributes[SpanAttributes.RECORD_ROOT.ERROR] == (
        "Incomplete hook turn"
    )


def test_tool_failure_does_not_fail_successful_turn():
    events = [
        parsers.parse_claude(_claude("UserPromptSubmit")),
        parsers.parse_claude(
            _claude(
                "PostToolUseFailure",
                tool_name="Bash",
                tool_use_id="tool-1",
                error="exit 1",
            )
        ),
        parsers.parse_claude(_claude("Stop")),
    ]

    spans = tracing.TraceAssembler().assemble(events)

    assert spans[0].status.status_code.name == "UNSET"
    tool = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.TOOL.value
    )
    assert tool.status.status_code.name == "ERROR"


class _Exporter:
    def __init__(self, result=SpanExportResult.SUCCESS):
        self.result = result
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return self.result


class _Session:
    def __init__(self, exporter):
        self.experimental_otel_exporter = exporter

    def force_flush(self):
        return True


class _FailedFlushSession(_Session):
    def force_flush(self):
        return False


def test_service_exports_terminal_turn_and_marks_it_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = _Exporter()
    event_journal = journal.EventJournal(tmp_path)
    hook_service = service.HookService(
        journal=event_journal,
        session=_Session(exporter),
    )

    assert hook_service.ingest(
        "cursor", _cursor("beforeSubmitPrompt", prompt="private")
    )
    assert hook_service.ingest("cursor", _cursor("stop", status="completed"))

    assert exporter.spans
    assert event_journal.pending_turns("cursor", "conversation-1") == []


def test_service_flush_retries_completed_turn_without_new_event(tmp_path: Path):
    event_journal = journal.EventJournal(tmp_path)
    event_journal.append(parsers.parse_cursor(_cursor("beforeSubmitPrompt")))
    event_journal.append(parsers.parse_cursor(_cursor("stop")))
    exporter = _Exporter(result=SpanExportResult.FAILURE)
    hook_service = service.HookService(
        journal=event_journal,
        session=_Session(exporter),
    )

    assert not hook_service.flush()
    event_journal.release_claim(
        "cursor", "conversation-1", "generation-1", failed=False
    )
    exporter.result = SpanExportResult.SUCCESS
    assert hook_service.flush()
    assert event_journal.pending_turns("cursor", "conversation-1") == []


def test_service_does_not_mark_exported_when_force_flush_fails(tmp_path: Path):
    exporter = _Exporter()
    event_journal = journal.EventJournal(tmp_path)
    hook_service = service.HookService(
        journal=event_journal,
        session=_FailedFlushSession(exporter),
    )

    assert hook_service.ingest("cursor", _cursor("beforeSubmitPrompt"))
    assert not hook_service.ingest("cursor", _cursor("stop"))
    assert event_journal.pending_turns("cursor", "conversation-1") == [
        "generation-1"
    ]


def test_trace_batch_can_use_standard_in_memory_exporter():
    events = [
        parsers.parse_claude(_claude("UserPromptSubmit")),
        parsers.parse_claude(_claude("Stop")),
    ]
    spans = tracing.TraceAssembler().assemble(events)
    exporter = InMemorySpanExporter()

    assert exporter.export(spans) == SpanExportResult.SUCCESS
    assert len(exporter.get_finished_spans()) == len(spans)


def test_snowflake_span_proto_preserves_error_status():
    events = [parsers.parse_claude(_claude("UserPromptSubmit"))]
    root = tracing.TraceAssembler().assemble(events, stale=True)[0]

    proto = exporter_utils.convert_readable_span_to_proto(root)

    assert proto.status.code == proto.status.STATUS_CODE_ERROR
    assert proto.status.message == ""
