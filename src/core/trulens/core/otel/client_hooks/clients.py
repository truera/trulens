"""Client plugin contracts and discovery for coding-agent hooks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

CLIENT_HOOK_ENTRY_POINT = "trulens.client_hooks"


@dataclass(frozen=True)
class FieldAliases:
    """Native payload field aliases consumed by the shared normalizer."""

    conversation: Tuple[str, ...] = (
        "conversation_id",
        "session_id",
        "sessionID",
    )
    turn: Tuple[str, ...] = (
        "generation_id",
        "turn_id",
        "prompt_id",
        "message_id",
        "messageID",
    )
    operation: Tuple[str, ...] = (
        "tool_use_id",
        "tool_call_id",
        "call_id",
        "callID",
        "operation_id",
        "subagent_id",
        "agent_id",
    )
    response: Tuple[str, ...] = (
        "response",
        "last_assistant_message",
        "text",
    )


@dataclass(frozen=True)
class ClientSpec:
    """Minimal native client contract supplied by plugin packages."""

    name: str
    aliases: Tuple[str, ...]
    user_config_path: Path
    project_config_path: Optional[Path]
    hook_events: Tuple[str, ...]
    field_aliases: FieldAliases = FieldAliases()
    config_builder: Optional[Callable[[str], Mapping[str, Any]]] = None
    plugin_builder: Optional[Callable[[str], str]] = None
    extract_overrides: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None

    def build_config(self, command: str) -> Mapping[str, Any]:
        """Build this client's native hook configuration fragment."""

        if self.config_builder is None:
            return {
                "hooks": {
                    event: [{"command": command}] for event in self.hook_events
                },
            }
        return self.config_builder(command)

    def build_plugin(self, command: str) -> Optional[str]:
        """Build a native plugin source file when this client is not JSON-hooks."""

        if self.plugin_builder is None:
            return None
        return self.plugin_builder(command)


_REGISTERED_CLIENTS: MutableMapping[str, ClientSpec] = {}


def register_client(spec: ClientSpec) -> ClientSpec:
    """Register a client specification and all of its aliases."""

    for name in (spec.name, *spec.aliases):
        _REGISTERED_CLIENTS[name] = spec
    return spec


def _load_plugins() -> None:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        discovered = entry_points.select(group=CLIENT_HOOK_ENTRY_POINT)
    else:
        discovered = entry_points.get(CLIENT_HOOK_ENTRY_POINT, ())
    for entry_point in discovered:
        loaded = entry_point.load()
        spec = (
            loaded()
            if callable(loaded) and not isinstance(loaded, ClientSpec)
            else loaded
        )
        if not isinstance(spec, ClientSpec):
            raise TypeError(
                f"Client hook plugin {entry_point.name} did not return ClientSpec."
            )
        register_client(spec)


def get_client(name: str) -> ClientSpec:
    """Return an installed client specification by name or alias."""

    _load_plugins()
    if name in _REGISTERED_CLIENTS:
        return _REGISTERED_CLIENTS[name]
    package = {
        "cursor": "trulens-apps-cursor",
        "claude": "trulens-apps-claude",
        "claude-code": "trulens-apps-claude",
        "opencode": "trulens-apps-opencode",
    }.get(name)
    suffix = f" Install it with: pip install {package}." if package else ""
    raise ValueError(f"Client '{name}' is not installed.{suffix}")


def list_clients() -> Tuple[ClientSpec, ...]:
    """List installed client specifications without duplicate aliases."""

    _load_plugins()
    return tuple(
        {spec.name: spec for spec in _REGISTERED_CLIENTS.values()}.values()
    )
