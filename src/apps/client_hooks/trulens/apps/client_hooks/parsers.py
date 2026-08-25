"""Parsers for Claude Code and Cursor command-hook payloads."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import hashlib
import json
from typing import Any, Mapping, Optional

from trulens.apps.client_hooks import models

_TERMINAL_EVENTS = {
    "stop",
    "stopfailure",
}
_FAILED_EVENTS = {"stopfailure", "posttoolusefailure"}
_START_EVENTS = {
    "pretooluse",
    "subagentstart",
    "beforeshellexecution",
    "beforemcpexecution",
}
_END_EVENTS = {
    "posttooluse",
    "posttoolusefailure",
    "subagentstop",
    "aftershellexecution",
    "aftermcpexecution",
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
    if "tool" in normalized or "shell" in normalized:
        return "tool"
    return "workflow"


def _phase(event_name: str) -> str:
    normalized = event_name.lower()
    if normalized in _START_EVENTS:
        return "start"
    if normalized in _END_EVENTS:
        return "end"
    return "instant"


def _parse(client: str, payload: Mapping[str, Any]) -> models.HookEvent:
    event_name = str(
        _first(payload, "hook_event_name", "event_name", "event") or "unknown"
    )
    normalized_name = event_name.lower()
    conversation_id = str(
        _first(payload, "session_id", "conversation_id") or "unknown"
    )
    turn_id = _first(
        payload,
        "generation_id",
        "turn_id",
        "prompt_id",
        "message_id",
    )
    operation_id = _first(
        payload,
        "tool_use_id",
        "tool_call_id",
        "operation_id",
        "subagent_id",
        "agent_id",
    )
    tool_name = _first(payload, "tool_name", "command_type")
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
        "conversation_id",
        "generation_id",
        "turn_id",
        "prompt_id",
        "message_id",
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
        model=_first(payload, "model", "model_id"),
        tool_name=str(tool_name) if tool_name is not None else None,
        server_name=str(server_name) if server_name is not None else None,
        duration_ms=_number(_first(payload, "duration_ms", "duration")),
        input_tokens=_integer(
            _first(payload, "input_tokens") or usage.get("input_tokens")
        ),
        output_tokens=_integer(
            _first(payload, "output_tokens") or usage.get("output_tokens")
        ),
        cost=_number(payload.get("cost")),
        prompt=payload.get("prompt"),
        response=_first(payload, "response", "last_assistant_message", "text"),
        tool_input=_first(payload, "tool_input", "command"),
        tool_output=_first(payload, "tool_output", "result_json", "output"),
        paths={
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


def parse(client: str, payload: Mapping[str, Any]) -> models.HookEvent:
    """Parse a supported client's hook payload."""

    if client == "claude":
        return parse_claude(payload)
    if client == "cursor":
        return parse_cursor(payload)
    raise ValueError(f"Unsupported client: {client}")
