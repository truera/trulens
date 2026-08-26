"""Minimal Claude Code-native hook specification."""

from pathlib import Path

from trulens.core.otel.client_hooks.clients import ClientSpec
from trulens.core.otel.client_hooks.clients import FieldAliases


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
)
