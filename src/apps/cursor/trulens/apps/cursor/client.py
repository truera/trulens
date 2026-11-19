"""Minimal Cursor-native hook specification."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from trulens.core.otel.client_hooks.clients import ClientSpec
from trulens.core.otel.client_hooks.clients import FieldAliases

_HOOK_EVENTS = (
    "beforeSubmitPrompt",
    "preToolUse",
    "postToolUse",
    "postToolUseFailure",
    "subagentStart",
    "subagentStop",
    "beforeShellExecution",
    "afterShellExecution",
    "beforeMCPExecution",
    "afterMCPExecution",
    "afterFileEdit",
    "afterAgentResponse",
    "stop",
)


def _config(command: str):
    return {
        "version": 1,
        "hooks": {event: [{"command": command}] for event in _HOOK_EVENTS},
    }


def _usage_value(usage: Mapping[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _is_human_prompt(entry: Mapping[str, Any]) -> bool:
    if entry.get("type") != "user":
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
        isinstance(part, Mapping) and part.get("type") == "text"
        for part in content
    )


def _transcript_overrides(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recover token usage from the transcript on the terminal `stop` event.

    Cursor's hook payloads never carry input/output token counts on any
    event (confirmed against Cursor's own hooks documentation) - unlike
    `model` (present directly on every hook) and response text (present on
    `afterAgentResponse` via the `text` field, already covered by this
    client's own `field_aliases.response`), neither of which need an
    override here.

    Mirrors `trulens.apps.claude.client._transcript_overrides`'s approach
    of reading the transcript file referenced in the hook payload and
    summing token usage for only the current (most recent) turn. Based on
    Cursor's documented `transcript_path` field plus third-party tooling
    that has independently reverse-engineered Cursor's transcript format as
    JSONL sharing Claude Code's `type`/`message.content[]`/`message.usage`
    shape (Cursor and Claude Code share the same underlying hook contract)
    - not verified against a live Cursor session. If the actual format
    differs, this fails soft (returns `{}`, same as any other parse error
    here) rather than raising, so worst case is no regression, not a crash.
    """
    if payload.get("hook_event_name") != "stop":
        return {}
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        return {}

    input_tokens = 0
    output_tokens = 0
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
                    input_tokens = 0
                    output_tokens = 0
                    continue
                if entry.get("type") != "assistant":
                    continue
                message = entry.get("message")
                if not isinstance(message, Mapping):
                    continue
                usage = message.get("usage")
                if not isinstance(usage, Mapping):
                    continue
                input_tokens += _usage_value(usage, "input_tokens")
                output_tokens += _usage_value(usage, "output_tokens")
    except OSError:
        return {}

    overrides: Dict[str, Any] = {}
    if input_tokens:
        overrides["input_tokens"] = input_tokens
    if output_tokens:
        overrides["output_tokens"] = output_tokens
    return overrides


client_spec = ClientSpec(
    name="cursor",
    aliases=(),
    user_config_path=Path("~/.cursor/hooks.json"),
    project_config_path=Path(".cursor/hooks.json"),
    hook_events=_HOOK_EVENTS,
    field_aliases=FieldAliases(
        conversation=("conversation_id",),
        turn=("generation_id",),
        operation=(
            "tool_call_id",
            "operation_id",
            "subagent_id",
            "agent_id",
        ),
        response=("response", "text"),
    ),
    config_builder=_config,
    extract_overrides=_transcript_overrides,
)
