"""Minimal Cursor-native hook specification."""

from pathlib import Path

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
)
