"""Parsers for Claude Code and Cursor command-hook payloads."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import hashlib
import json
from typing import Any, Mapping, Optional

from trulens.core.otel.client_hooks import clients
from trulens.core.otel.client_hooks import models

_TERMINAL_EVENTS = {
    "stop",
    "stopfailure",
    "session.idle",
    "session.error",
    "sessionend",
}
_FAILED_EVENTS = {
    "stopfailure",
    "posttoolusefailure",
    "session.error",
}
_START_EVENTS = {
    "pretooluse",
    "subagentstart",
    "beforeshellexecution",
    "beforemcpexecution",
    "tool.execute.before",
}
_END_EVENTS = {
    "posttooluse",
    "posttoolusefailure",
    "subagentstop",
    "aftershellexecution",
    "aftermcpexecution",
    "tool.execute.after",
    "file.edited",
}


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _integer(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fingerprint(client: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, default=str, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(client.encode() + b":" + encoded).hexdigest()


def _timestamp(payload: Mapping[str, Any]) -> datetime:
    value = _first(payload, "timestamp", "observed_at", "created_at")
    if value is None:
        return models.utc_now()
    if isinstance(value, (int, float)):
        divisor = 1000 if value > 10_000_000_000 else 1
        return datetime.fromtimestamp(value / divisor, timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return models.utc_now()
    return models.utc_now()


def _category(event_name: str, tool_name: Optional[str]) -> str:
    normalized = event_name.lower()
    if "subagent" in normalized:
        return "agent"
    if "mcp" in normalized or (tool_name or "").lower().startswith("mcp"):
        return "mcp"
    if (
        "tool" in normalized
        or "shell" in normalized
        or "fileedit" in normalized
        or "file.edit" in normalized
    ):
        return "tool"
    return "workflow"


def _phase(event_name: str) -> str:
    normalized = event_name.lower()
    if normalized in _START_EVENTS:
        return "start"
    if normalized in _END_EVENTS:
        return "end"
    return "instant"


def _diff(payload: Mapping[str, Any]) -> Any:
    explicit_diff = _first(payload, "diff", "patch")
    if explicit_diff is not None:
        return explicit_diff
    edits = payload.get("edits")
    if edits is not None:
        return {
            "file_path": payload.get("file_path"),
            "edits": edits,
        }
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, Mapping):
        return _first(tool_input, "diff", "patch")
    return None


def _parse(
    client: str,
    payload: Mapping[str, Any],
    spec: Optional[clients.ClientSpec] = None,
) -> models.HookEvent:
    aliases = spec.field_aliases if spec is not None else clients.FieldAliases()
    overrides = (
        spec.extract_overrides(payload)
        if spec is not None and spec.extract_overrides is not None
        else {}
    )
    raw_event = overrides.get("event_name") or _first(
        payload, "hook_event_name", "event_name", "event"
    )
    event_name = (
        str(raw_event)
        if raw_event is not None and not isinstance(raw_event, Mapping)
        else "unknown"
    )
    normalized_name = event_name.lower()
    conversation_id = overrides.get("conversation_id") or _first(
        payload, *aliases.conversation
    )
    if conversation_id is None:
        raise ValueError(f"{client} hook payload is missing a conversation ID.")
    conversation_id = str(conversation_id)
    turn_id = overrides.get("turn_id") or _first(payload, *aliases.turn)
    operation_id = overrides.get("operation_id") or _first(
        payload, *aliases.operation
    )
    tool_name = overrides.get("tool_name") or _first(
        payload, "tool_name", "command_type", "tool"
    )
    server_name = _first(payload, "mcp_server_name", "server_name")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    failed = normalized_name in _FAILED_EVENTS or str(
        payload.get("status", "")
    ).lower() in {"error", "failed"}
    event_id = str(
        _first(payload, "event_id", "hook_id")
        or f"{event_name}:{operation_id or _fingerprint(client, payload)}"
    )
    known_keys = {
        "session_id",
        "sessionID",
        "conversation_id",
        "generation_id",
        "turn_id",
        "prompt_id",
        "message_id",
        "messageID",
        "call_id",
        "callID",
        "tool",
        "args",
        "parts",
        "hook_event_name",
        "event_name",
        "event",
        "event_id",
        "hook_id",
        "tool_use_id",
        "tool_call_id",
        "operation_id",
        "subagent_id",
        "agent_id",
        "tool_name",
        "command_type",
        "mcp_server_name",
        "server_name",
        "model",
        "model_id",
        "duration_ms",
        "duration",
        "input_tokens",
        "output_tokens",
        "cost",
        "usage",
        "prompt",
        "response",
        "last_assistant_message",
        "text",
        "tool_input",
        "tool_output",
        "tool_response",
        "diff",
        "patch",
        "edits",
        "file_path",
        "result_json",
        "output",
        "command",
        "cwd",
        "workspace_roots",
        "transcript_path",
        "error",
        "status",
        "timestamp",
        "observed_at",
        "created_at",
    }
    metadata = {
        key: value for key, value in payload.items() if key not in known_keys
    }
    return models.HookEvent(
        client=client,
        event_name=event_name,
        event_id=event_id,
        conversation_id=conversation_id,
        turn_id=str(turn_id) if turn_id is not None else None,
        operation_id=str(operation_id) if operation_id is not None else None,
        observed_at=_timestamp(payload),
        phase=_phase(event_name),
        category=_category(event_name, str(tool_name) if tool_name else None),
        terminal=normalized_name in _TERMINAL_EVENTS,
        failed=failed,
        model=overrides.get("model") or _first(payload, "model", "model_id"),
        tool_name=str(tool_name) if tool_name is not None else None,
        server_name=str(server_name) if server_name is not None else None,
        duration_ms=_number(_first(payload, "duration_ms", "duration")),
        input_tokens=_integer(
            overrides.get("input_tokens")
            or _first(payload, "input_tokens")
            or usage.get("input_tokens")
        ),
        output_tokens=_integer(
            overrides.get("output_tokens")
            or _first(payload, "output_tokens")
            or usage.get("output_tokens")
        ),
        cost=_number(payload.get("cost")),
        prompt=overrides.get("prompt", payload.get("prompt")),
        response=overrides.get("response")
        or _first(payload, *aliases.response),
        tool_input=overrides.get("tool_input")
        or _first(payload, "tool_input", "command", "args"),
        tool_output=overrides.get("tool_output")
        or _first(
            payload, "tool_output", "tool_response", "result_json", "output"
        ),
        diff=_diff(payload),
        paths={
            "file_path": payload.get("file_path"),
            "cwd": payload.get("cwd"),
            "workspace_roots": payload.get("workspace_roots"),
            "transcript_path": payload.get("transcript_path"),
        },
        error=str(payload["error"]) if payload.get("error") else None,
        metadata=metadata,
    )


def parse_claude(payload: Mapping[str, Any]) -> models.HookEvent:
    """Parse one Claude Code hook payload."""

    return _parse("claude", payload)


def parse_cursor(payload: Mapping[str, Any]) -> models.HookEvent:
    """Parse one Cursor hook payload."""

    return _parse("cursor", payload)


def parse_opencode(payload: Mapping[str, Any]) -> models.HookEvent:
    """Parse one OpenCode plugin hook payload."""

    spec = None
    try:
        spec = clients.get_client("opencode")
    except ValueError:
        spec = None
    return _parse("opencode", payload, spec=spec)


def parse(client: str, payload: Mapping[str, Any]) -> models.HookEvent:
    """Parse a supported client's hook payload."""

    try:
        spec = clients.get_client(client)
    except ValueError:
        if client == "claude":
            return parse_claude(payload)
        if client == "cursor":
            return parse_cursor(payload)
        if client == "opencode":
            return parse_opencode(payload)
        raise
    return _parse(spec.name, payload, spec=spec)
