"""Minimal Claude Code-native hook specification."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from trulens.core.otel.client_hooks.clients import ClientSpec
from trulens.core.otel.client_hooks.clients import FieldAliases


def _usage_value(usage: Mapping[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _is_human_prompt(entry: Mapping[str, Any]) -> bool:
    if entry.get("type") != "user" or entry.get("isSidechain"):
        return False
    message = entry.get("message")
    if not isinstance(message, Mapping):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, Mapping)
        and part.get("type") in {"text", "image", "document"}
        for part in content
    )


def _transcript_overrides(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("hook_event_name") != "Stop":
        return {}
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        return {}

    messages: Dict[str, Mapping[str, Any]] = {}
    response_parts: Dict[str, list[str]] = {}
    response_order: list[str] = []
    try:
        with Path(str(transcript_path)).open(encoding="utf-8") as transcript:
            for line in transcript:
                try:
                    entry = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(entry, Mapping):
                    continue
                if _is_human_prompt(entry):
                    messages.clear()
                    response_parts.clear()
                    response_order.clear()
                    continue
                if entry.get("isSidechain") or entry.get("type") != "assistant":
                    continue
                message = entry.get("message")
                if not isinstance(message, Mapping):
                    continue
                message_id = message.get("id")
                if not message_id:
                    continue
                message_id = str(message_id)
                if message_id in messages:
                    del messages[message_id]
                messages[message_id] = message
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                text_parts = [
                    str(part["text"])
                    for part in content
                    if isinstance(part, Mapping)
                    and part.get("type") == "text"
                    and part.get("text")
                ]
                if text_parts:
                    response_parts[message_id] = text_parts
                    if message_id in response_order:
                        response_order.remove(message_id)
                    response_order.append(message_id)
    except OSError:
        return {}

    input_tokens = 0
    output_tokens = 0
    for message in messages.values():
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_tokens += _usage_value(usage, "input_tokens")
        input_tokens += _usage_value(usage, "cache_creation_input_tokens")
        input_tokens += _usage_value(usage, "cache_read_input_tokens")
        output_tokens += _usage_value(usage, "output_tokens")

    overrides: Dict[str, Any] = {}
    if messages:
        last_message = next(reversed(messages.values()))
        if last_message.get("model"):
            overrides["model"] = last_message["model"]
    if input_tokens:
        overrides["input_tokens"] = input_tokens
    if output_tokens:
        overrides["output_tokens"] = output_tokens
    if response_order:
        overrides["response"] = "\n".join(response_parts[response_order[-1]])
    return overrides


def _config(command: str):
    events = (
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    )
    return {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": command}]}]
            for event in events
        }
    }


client_spec = ClientSpec(
    name="claude-code",
    aliases=("claude",),
    user_config_path=Path("~/.claude/settings.json"),
    project_config_path=Path(".claude/settings.json"),
    hook_events=(
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    ),
    field_aliases=FieldAliases(
        conversation=("session_id",),
        turn=("turn_id", "prompt_id", "message_id"),
        operation=("tool_use_id", "subagent_id", "agent_id"),
        response=("last_assistant_message", "response"),
    ),
    config_builder=_config,
    extract_overrides=_transcript_overrides,
)
