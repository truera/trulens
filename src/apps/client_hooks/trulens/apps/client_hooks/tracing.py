"""Assemble complete OpenTelemetry traces from hook event journals."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from trulens.apps.client_hooks import models
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes


def _otel_id(seed: str, bits: int) -> int:
    size = bits // 8
    value = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:size])
    return value or 1


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _nanoseconds(event: models.HookEvent) -> int:
    return int(event.observed_at.timestamp() * 1_000_000_000)


def _duration_end(event: models.HookEvent) -> int:
    if event.duration_ms is None:
        return _nanoseconds(event)
    return _nanoseconds(event) + int(event.duration_ms * 1_000_000)


def _pair_events(
    events: Iterable[models.HookEvent],
) -> List[Tuple[models.HookEvent, models.HookEvent]]:
    pending: Dict[Tuple[str, str, str], models.HookEvent] = {}
    pairs: List[Tuple[models.HookEvent, models.HookEvent]] = []
    for event in events:
        key = (
            event.category,
            event.operation_id
            or event.tool_name
            or event.server_name
            or event.category,
            event.tool_name or event.server_name or "",
        )
        if event.phase == "start":
            pending[key] = event
        elif event.phase == "end" and key in pending:
            pairs.append((pending.pop(key), event))
        else:
            pairs.append((event, event))
    pairs.extend((event, event) for event in pending.values())
    return pairs


def _span_type(category: str) -> SpanAttributes.SpanType:
    return {
        "agent": SpanAttributes.SpanType.AGENT,
        "mcp": SpanAttributes.SpanType.MCP,
        "tool": SpanAttributes.SpanType.TOOL,
        "workflow": SpanAttributes.SpanType.WORKFLOW_STEP,
    }.get(category, SpanAttributes.SpanType.UNKNOWN)


class TraceAssembler:
    """Convert one completed client turn into TruLens-compatible spans."""

    def __init__(
        self,
        *,
        app_name: str = "coding-agent",
        app_version: str = "client-hooks",
    ) -> None:
        self.app_name = app_name
        self.app_version = app_version

    def assemble(
        self, events: List[models.HookEvent], *, stale: bool = False
    ) -> List[ReadableSpan]:
        """Assemble root, agent, and operation spans for one turn."""

        if not events:
            return []
        events = sorted(events, key=lambda event: event.observed_at)
        first = events[0]
        last = events[-1]
        turn_id = first.turn_id or first.event_id
        record_id = f"{first.client}:{first.conversation_id}:{turn_id}"
        trace_id = _otel_id(f"trace:{record_id}", 128)
        root_span_id = _otel_id(f"root:{record_id}", 64)
        agent_span_id = _otel_id(f"agent:{record_id}", 64)
        common = {
            ResourceAttributes.APP_NAME: self.app_name,
            ResourceAttributes.APP_VERSION: self.app_version,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.CONVERSATION_ID: first.conversation_id,
            SpanAttributes.INPUT_ID: turn_id,
        }
        prompt = next((event.prompt for event in events if event.prompt), None)
        response = next(
            (event.response for event in reversed(events) if event.response),
            None,
        )
        terminal_failure = next(
            (
                event
                for event in reversed(events)
                if event.terminal and event.failed
            ),
            None,
        )
        root_failed = stale or terminal_failure is not None
        root_error = (
            "Incomplete hook turn"
            if stale
            else terminal_failure.error
            if terminal_failure is not None
            else None
        )
        root_attributes: Dict[str, Any] = {
            **common,
            SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT.value,
            SpanAttributes.RECORD_ROOT.INPUT: prompt
            or "[content not captured]",
            SpanAttributes.RECORD_ROOT.OUTPUT: response
            or ("[incomplete]" if stale else "[content not captured]"),
            SpanAttributes.CALL.FUNCTION: f"{first.client}.turn",
            SpanAttributes.WORKFLOW.AGENT_NAME: first.client,
        }
        if root_error:
            root_attributes[SpanAttributes.RECORD_ROOT.ERROR] = root_error
        spans = [
            self._span(
                name=f"{first.client}.turn",
                trace_id=trace_id,
                span_id=root_span_id,
                parent_id=None,
                attributes=root_attributes,
                start_time=_nanoseconds(first),
                end_time=max(_nanoseconds(last), _duration_end(last)),
                failed=root_failed,
            )
        ]
        agent_attributes = {
            **common,
            SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.AGENT.value,
            SpanAttributes.CALL.FUNCTION: first.client,
            SpanAttributes.WORKFLOW.AGENT_NAME: first.client,
        }
        model = next((event.model for event in events if event.model), None)
        if model:
            agent_attributes[SpanAttributes.COST.MODEL] = model
            agent_attributes[GenAIAttributes.REQUEST.MODEL] = model
            agent_attributes[GenAIAttributes.SYSTEM.NAME] = (
                "anthropic" if first.client == "claude" else "cursor"
            )
        input_tokens = sum(event.input_tokens or 0 for event in events)
        output_tokens = sum(event.output_tokens or 0 for event in events)
        if input_tokens or output_tokens:
            agent_attributes[SpanAttributes.COST.NUM_PROMPT_TOKENS] = (
                input_tokens
            )
            agent_attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = (
                output_tokens
            )
            agent_attributes[SpanAttributes.COST.NUM_TOKENS] = (
                input_tokens + output_tokens
            )
            agent_attributes[GenAIAttributes.USAGE.INPUT_TOKENS] = input_tokens
            agent_attributes[GenAIAttributes.USAGE.OUTPUT_TOKENS] = (
                output_tokens
            )
        cost = sum(event.cost or 0.0 for event in events)
        if cost:
            agent_attributes[SpanAttributes.COST.COST] = cost
            agent_attributes[SpanAttributes.COST.CURRENCY] = "USD"
        spans.append(
            self._span(
                name=f"{first.client}.agent",
                trace_id=trace_id,
                span_id=agent_span_id,
                parent_id=root_span_id,
                attributes=agent_attributes,
                start_time=_nanoseconds(first),
                end_time=max(_nanoseconds(last), _duration_end(last)),
                failed=root_failed,
            )
        )
        for index, (start, finish) in enumerate(_pair_events(events)):
            if start.category == "workflow" and (
                start.event_name.lower()
                in {"userpromptsubmit", "beforesubmitprompt", "stop"}
            ):
                continue
            spans.append(
                self._operation_span(
                    start=start,
                    finish=finish,
                    index=index,
                    trace_id=trace_id,
                    parent_id=agent_span_id,
                    common=common,
                )
            )
        return spans

    def _operation_span(
        self,
        *,
        start: models.HookEvent,
        finish: models.HookEvent,
        index: int,
        trace_id: int,
        parent_id: int,
        common: Mapping[str, Any],
    ) -> ReadableSpan:
        name = start.tool_name or start.server_name or start.event_name
        attributes: Dict[str, Any] = {
            **common,
            SpanAttributes.SPAN_TYPE: _span_type(start.category).value,
            SpanAttributes.CALL.FUNCTION: name,
            "coding_agent.client": start.client,
            "coding_agent.hook_event": finish.event_name,
        }
        if start.tool_name:
            attributes[GenAIAttributes.TOOL.NAME] = start.tool_name
        if start.tool_input is not None:
            attributes[GenAIAttributes.TOOL.CALL_ARGUMENTS] = _json(
                start.tool_input
            )
            attributes[f"{SpanAttributes.CALL.KWARGS}.input"] = _json(
                start.tool_input
            )
        if finish.tool_output is not None:
            attributes[GenAIAttributes.TOOL.CALL_RESULT] = _json(
                finish.tool_output
            )
            attributes[SpanAttributes.CALL.RETURN] = _json(finish.tool_output)
        if start.category == "mcp":
            if start.tool_name:
                attributes[SpanAttributes.MCP.TOOL_NAME] = start.tool_name
            if start.server_name:
                attributes[SpanAttributes.MCP.SERVER_NAME] = start.server_name
            if start.tool_input is not None:
                attributes[SpanAttributes.MCP.INPUT_ARGUMENTS] = _json(
                    start.tool_input
                )
            if finish.tool_output is not None:
                attributes[SpanAttributes.MCP.OUTPUT_CONTENT] = _json(
                    finish.tool_output
                )
            attributes[SpanAttributes.MCP.OUTPUT_IS_ERROR] = finish.failed
            if finish.duration_ms is not None:
                attributes[SpanAttributes.MCP.EXECUTION_TIME_MS] = (
                    finish.duration_ms
                )
        if finish.error:
            attributes[SpanAttributes.CALL.ERROR] = finish.error
        start_time = _nanoseconds(start)
        if finish is start:
            end_time = _nanoseconds(finish)
            if finish.duration_ms is not None:
                start_time = end_time - int(finish.duration_ms * 1_000_000)
        else:
            end_time = _nanoseconds(finish)
        return self._span(
            name=name,
            trace_id=trace_id,
            span_id=_otel_id(
                f"operation:{trace_id}:{start.operation_id or start.event_id}:{index}",
                64,
            ),
            parent_id=parent_id,
            attributes=attributes,
            start_time=start_time,
            end_time=end_time,
            failed=finish.failed,
            kind=trace.SpanKind.CLIENT
            if start.category == "mcp"
            else trace.SpanKind.INTERNAL,
        )

    @staticmethod
    def _span(
        *,
        name: str,
        trace_id: int,
        span_id: int,
        parent_id: Optional[int],
        attributes: Mapping[str, Any],
        start_time: int,
        end_time: int,
        failed: bool,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ) -> ReadableSpan:
        context = trace.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace.TraceFlags.SAMPLED,
        )
        parent = None
        if parent_id is not None:
            parent = trace.SpanContext(
                trace_id=trace_id,
                span_id=parent_id,
                is_remote=False,
                trace_flags=trace.TraceFlags.SAMPLED,
            )
        return ReadableSpan(
            name=name,
            context=context,
            parent=parent,
            resource=Resource.create({"service.name": "trulens-client-hooks"}),
            attributes=attributes,
            kind=kind,
            status=Status(StatusCode.ERROR if failed else StatusCode.UNSET),
            start_time=start_time,
            end_time=max(start_time, end_time),
        )


def stale_after_from_environment() -> timedelta:
    """Read the incomplete-turn timeout from adapter configuration."""

    import os

    raw_hours = os.environ.get("TRULENS_HOOKS_STALE_AFTER_HOURS", "24")
    try:
        hours = max(1.0 / 60.0, float(raw_hours))
    except ValueError:
        hours = 24
    return timedelta(hours=hours)
