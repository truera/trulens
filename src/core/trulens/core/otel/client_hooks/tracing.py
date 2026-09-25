"""Assemble complete OpenTelemetry traces from hook event journals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from trulens.core.otel.client_hooks import models
from trulens.core.otel.propagation import extract_trace_context
from trulens.otel.semconv.trace import ErrorAttributes
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import GenAIEvents
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

try:
    _TRULENS_VERSION = _package_version("trulens-core")
except PackageNotFoundError:
    _TRULENS_VERSION = "unknown"

# Turn spans are built as completed ReadableSpans rather than through a Tracer,
# and nothing else stamps an instrumentation scope on a span, so without this
# they arrive with an empty scope. Consumers then have no scope name to filter
# these spans by, which matters most when they share a trace with other
# producers that do set one.
_INSTRUMENTATION_SCOPE = InstrumentationScope(
    "trulens.client_hooks", _TRULENS_VERSION
)


def _otel_id(seed: str, bits: int) -> int:
    size = bits // 8
    # byteorder is required on Python 3.10 (it only became optional, defaulting
    # to "big", in 3.11), so pass it explicitly. "big" matches the 3.11+ default,
    # keeping ids unchanged on newer versions.
    value = int.from_bytes(
        hashlib.sha256(seed.encode()).digest()[:size], byteorder="big"
    )
    return value or 1


def environ_trace_context(
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Read the W3C trace context the surrounding process exported, if any.

    A coding agent that propagates trace context sets `TRACEPARENT`, and
    sometimes `TRACESTATE`, on the processes it spawns. That is how a hook
    invocation learns which span it was spawned under, since the payload on
    stdin carries no trace context of its own.

    The values are returned as read and validated later, when spans are
    assembled, so a value that does not parse stays visible in the journal
    instead of being dropped at capture time.

    Args:
        environ: Environment mapping to read. Defaults to `os.environ`.

    Returns:
        The `(traceparent, tracestate)` pair, with None for anything absent.
        `tracestate` is only returned alongside a `traceparent`, since it
        carries no meaning on its own.
    """

    source = os.environ if environ is None else environ
    traceparent = source.get("TRACEPARENT") or None
    if traceparent is None:
        return None, None
    return traceparent, source.get("TRACESTATE") or None


def inherited_span_context(
    events: Iterable[models.HookEvent],
) -> Optional[trace.SpanContext]:
    """Return the remote span context a turn's events were spawned under.

    Hooks fire repeatedly across one turn, and only some of them run while the
    agent holds an active span, so the events carry the context unevenly. The
    earliest one that parses wins, which puts assembled spans under the
    outermost span the agent had open rather than under a later, narrower one.

    Args:
        events: The turn's events, in any order.

    Returns:
        A remote `SpanContext` when at least one event carries a valid
        `traceparent`, otherwise None.
    """

    for event in sorted(events, key=lambda event: event.observed_at):
        if not event.traceparent:
            continue
        carrier = {"traceparent": event.traceparent}
        if event.tracestate:
            carrier["tracestate"] = event.tracestate
        span_context = extract_trace_context(carrier)
        if span_context is not None:
            return span_context
    return None


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _message(role: str, content: str) -> Dict[str, Any]:
    """Build an OTEL GenAI structured text message."""

    return {
        "role": role,
        "parts": [{"type": "text", "content": content}],
    }


def _provider_for_model(model: str) -> Optional[str]:
    normalized = model.lower()
    if normalized.startswith("claude") or "anthropic/" in normalized:
        return "anthropic"
    if (
        normalized.startswith(("gpt-", "o1", "o3", "o4"))
        or "openai/" in normalized
    ):
        return "openai"
    return None


def _reported_usage(events: Iterable[models.HookEvent], attribute: str) -> int:
    """Use the latest reported usage to avoid summing cumulative hook values."""

    values = [getattr(event, attribute) for event in events]
    return next((value for value in reversed(values) if value is not None), 0)


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


@dataclass(frozen=True)
class TurnIdentity:
    """Run and record identity for one assembled coding-agent turn.

    A turn is the unit of export and the unit of ingestion: each turn
    contributes exactly one record to its run. ``run_name`` is shared by every
    turn in a conversation, so a conversation maps to one run carrying one
    completed invocation per turn.
    """

    client: str
    conversation_id: str
    turn_id: str
    record_id: str
    app_name: str
    app_version: str
    run_name: str
    input_records_count: int = 1


class TraceAssembler:
    """Convert one completed client turn into TruLens-compatible spans."""

    def __init__(
        self,
        *,
        app_name: Optional[str] = None,
        app_version: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.app_name = app_name
        self.app_version = app_version
        self.run_name = run_name

    def identify(
        self, events: List[models.HookEvent]
    ) -> Optional[TurnIdentity]:
        """Resolve run and record identity for one turn.

        Returns ``None`` for an empty turn. Identity is derived from the native
        client payloads so that it stays stable across retries: re-exporting a
        turn reuses the same record and run names rather than creating new ones.
        """

        if not events:
            return None
        events = sorted(events, key=lambda event: event.observed_at)
        first = events[0]
        turn_id = first.turn_id or first.event_id
        app_name = self.app_name or first.client.removesuffix("-code")
        app_version = (
            self.app_version
            or next(
                (
                    event.metadata.get("cursor_version")
                    or event.metadata.get("client_version")
                    for event in reversed(events)
                    if event.metadata.get("cursor_version")
                    or event.metadata.get("client_version")
                ),
                None,
            )
            or "unknown"
        )
        return TurnIdentity(
            client=first.client,
            conversation_id=first.conversation_id,
            turn_id=turn_id,
            record_id=f"{first.client}:{first.conversation_id}:{turn_id}",
            app_name=app_name,
            app_version=app_version,
            run_name=self.run_name or first.conversation_id,
        )

    def assemble(
        self, events: List[models.HookEvent], *, stale: bool = False
    ) -> List[ReadableSpan]:
        """Assemble root, agent, and operation spans for one turn."""

        if not events:
            return []
        events = sorted(events, key=lambda event: event.observed_at)
        first = events[0]
        last = events[-1]
        identity = self.identify(events)
        turn_id = identity.turn_id
        app_name = identity.app_name
        app_version = identity.app_version
        run_name = identity.run_name
        record_id = identity.record_id
        # When the agent handed this turn's hooks a trace context, adopt its
        # trace so these spans land beside the agent's own spans and whatever
        # the gateway recorded for the same requests. Without one, the turn
        # keeps its self-rooted trace derived from the record id.
        inherited = inherited_span_context(events)
        trace_id = (
            inherited.trace_id
            if inherited is not None
            else _otel_id(f"trace:{record_id}", 128)
        )
        agent_span_id = _otel_id(f"agent:{record_id}", 64)
        root_span_id = _otel_id(f"root:{record_id}", 64)
        response_span_id = _otel_id(f"response:{record_id}", 64)
        common = {
            ResourceAttributes.APP_NAME: app_name,
            ResourceAttributes.APP_VERSION: app_version,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.CONVERSATION_ID: first.conversation_id,
            # The GenAI-standard spelling of the same value, so these spans can
            # be joined to producers outside TruLens that record the server side
            # of the same conversation.
            GenAIAttributes.CONVERSATION.ID: first.conversation_id,
            SpanAttributes.INPUT_ID: turn_id,
            SpanAttributes.RUN_NAME: run_name,
            SpanAttributes.INPUT_RECORDS_COUNT: identity.input_records_count,
        }
        prompt = next((event.prompt for event in events if event.prompt), None)
        response = next(
            (event.response for event in reversed(events) if event.response),
            None,
        )
        response_event = next(
            (event for event in reversed(events) if event.response),
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
        if response is not None:
            root_attributes[SpanAttributes.CALL.RETURN] = _json(response)
        if root_failed:
            if root_error:
                root_attributes[SpanAttributes.RECORD_ROOT.ERROR] = root_error
            root_attributes[ErrorAttributes.TYPE] = (
                "incomplete_turn" if stale else "client_hook_error"
            )
        agent_attributes = {
            **common,
            SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.AGENT.value,
            SpanAttributes.CALL.FUNCTION: first.client,
            SpanAttributes.WORKFLOW.AGENT_NAME: first.client,
        }
        request_model = next(
            (event.model for event in events if event.model), None
        )
        response_model = (
            response_event.model if response_event is not None else None
        )
        if request_model:
            agent_attributes[SpanAttributes.COST.MODEL] = request_model
        input_tokens = _reported_usage(events, "input_tokens")
        output_tokens = _reported_usage(events, "output_tokens")
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
        cost = sum(event.cost or 0.0 for event in events)
        if cost:
            agent_attributes[SpanAttributes.COST.COST] = cost
            agent_attributes[SpanAttributes.COST.CURRENCY] = "USD"
        spans = [
            self._span(
                name=f"{first.client}.request_response",
                trace_id=trace_id,
                span_id=root_span_id,
                parent_id=None,
                parent_context=inherited,
                attributes=root_attributes,
                start_time=_nanoseconds(first),
                end_time=max(_nanoseconds(last), _duration_end(last)),
                failed=root_failed,
                status_description=root_error,
            )
        ]
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
                status_description=root_error,
            )
        )
        if response is not None:
            response_attributes = {
                **common,
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.GENERATION.value,
                SpanAttributes.CALL.FUNCTION: "response_generation",
                SpanAttributes.CALL.RETURN: _json(response),
                SpanAttributes.RECORD_ROOT.OUTPUT: response,
                GenAIAttributes.OPERATION.NAME: "chat",
            }
            if request_model:
                response_attributes[GenAIAttributes.REQUEST.MODEL] = (
                    request_model
                )
                provider = _provider_for_model(request_model)
                if provider:
                    response_attributes[GenAIAttributes.SYSTEM.NAME] = provider
            if response_model:
                response_attributes[GenAIAttributes.RESPONSE.MODEL] = (
                    response_model
                )
            if input_tokens or output_tokens:
                response_attributes[GenAIAttributes.USAGE.INPUT_TOKENS] = (
                    input_tokens
                )
                response_attributes[GenAIAttributes.USAGE.OUTPUT_TOKENS] = (
                    output_tokens
                )
            response_event_attributes = {
                GenAIEvents.EventAttributes.OUTPUT_MESSAGES: _json([
                    _message("assistant", response)
                ])
            }
            if prompt is not None:
                response_event_attributes[
                    GenAIEvents.EventAttributes.INPUT_MESSAGES
                ] = _json([_message("user", prompt)])
            spans.append(
                self._span(
                    name=f"chat {request_model}" if request_model else "chat",
                    trace_id=trace_id,
                    span_id=response_span_id,
                    parent_id=agent_span_id,
                    attributes=response_attributes,
                    start_time=_nanoseconds(first),
                    end_time=_nanoseconds(response_event),
                    failed=False,
                    kind=trace.SpanKind.CLIENT,
                    event_attributes=response_event_attributes,
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
        tool_name = start.tool_name or start.server_name or start.event_name
        is_tool_operation = start.category in {"tool", "mcp"}
        name = f"execute_tool {tool_name}" if is_tool_operation else tool_name
        attributes: Dict[str, Any] = {
            **common,
            SpanAttributes.SPAN_TYPE: _span_type(start.category).value,
            SpanAttributes.CALL.FUNCTION: tool_name,
            SpanAttributes.CODING_AGENT.CLIENT: start.client,
            SpanAttributes.CODING_AGENT.NATIVE_EVENT: finish.event_name,
        }
        if is_tool_operation:
            attributes[GenAIAttributes.OPERATION.NAME] = "execute_tool"
            if start.operation_id:
                attributes[GenAIAttributes.TOOL.CALL_ID] = start.operation_id
        editor_version = finish.metadata.get(
            "cursor_version"
        ) or start.metadata.get("cursor_version")
        if editor_version:
            attributes[SpanAttributes.CODING_AGENT.EDITOR_VERSION] = (
                editor_version
            )
        workspace = None
        if finish.paths:
            workspace = finish.paths.get("workspace_roots") or finish.paths.get(
                "cwd"
            )
        if workspace is None and start.paths:
            workspace = start.paths.get("workspace_roots") or start.paths.get(
                "cwd"
            )
        if workspace:
            attributes[SpanAttributes.CODING_AGENT.WORKSPACE] = _json(workspace)
        if is_tool_operation:
            attributes[GenAIAttributes.TOOL.NAME] = tool_name
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
        if finish.response is not None:
            attributes[SpanAttributes.CALL.RETURN] = _json(finish.response)
        diff = finish.diff if finish.diff is not None else start.diff
        if diff is not None:
            serialized_diff = _json(diff)
            attributes[SpanAttributes.CODING_AGENT.DIFF] = serialized_diff
            attributes[SpanAttributes.CALL.RETURN] = serialized_diff
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
            attributes[ErrorAttributes.TYPE] = "client_hook_error"
        elif finish.failed:
            attributes[ErrorAttributes.TYPE] = "client_hook_error"
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
            status_description=finish.error,
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
        event_attributes: Optional[Mapping[str, Any]] = None,
        status_description: Optional[str] = None,
        parent_context: Optional[trace.SpanContext] = None,
    ) -> ReadableSpan:
        context = trace.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace.TraceFlags.SAMPLED,
        )
        # A remote parent arrives as a fully formed context and is used as is,
        # so its is_remote flag and trace flags survive. Parents inside this
        # batch are built from a local span id instead.
        parent = parent_context
        if parent is None and parent_id is not None:
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
            instrumentation_scope=_INSTRUMENTATION_SCOPE,
            resource=Resource.create({
                "service.name": attributes.get(
                    ResourceAttributes.APP_NAME, "trulens-client-hooks"
                ),
                "service.version": attributes.get(
                    ResourceAttributes.APP_VERSION, "unknown"
                ),
            }),
            attributes=attributes,
            events=(
                Event(
                    GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                    attributes=event_attributes,
                    timestamp=end_time,
                ),
            )
            if event_attributes
            else (),
            kind=kind,
            status=Status(
                StatusCode.ERROR if failed else StatusCode.UNSET,
                status_description if failed else None,
            ),
            start_time=start_time,
            end_time=max(start_time, end_time),
        )


def stale_after_from_environment() -> timedelta:
    """Read the incomplete-turn timeout from adapter configuration."""

    import os

    raw_hours = os.environ.get("TRULENS_STALE_AFTER_HOURS", "24")
    try:
        hours = max(1.0 / 60.0, float(raw_hours))
    except ValueError:
        hours = 24
    return timedelta(hours=hours)
