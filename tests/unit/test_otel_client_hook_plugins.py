"""Contract tests for thin coding-agent client plugins."""

import json

import pytest
from trulens.apps.claude import client_spec as claude_spec
from trulens.apps.cursor import client_spec as cursor_spec
from trulens.apps.opencode import client_spec as opencode_spec
from trulens.core.otel.client_hooks import cli
from trulens.core.otel.client_hooks import clients
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import tracing
from trulens.otel.semconv.trace import SpanAttributes


def test_cursor_plugin_is_declarative():
    assert cursor_spec.name == "cursor"
    assert "afterFileEdit" in cursor_spec.hook_events
    assert cursor_spec.field_aliases.conversation == ("conversation_id",)


def test_claude_plugin_is_declarative():
    assert claude_spec.name == "claude-code"
    assert "claude" in claude_spec.aliases
    assert claude_spec.field_aliases.conversation == ("session_id",)
    assert claude_spec.hook_events == (
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    )
    assert "StopFailure" not in claude_spec.build_config("hook")["hooks"]


def test_opencode_plugin_is_declarative():
    assert opencode_spec.name == "opencode"
    assert "open-code" in opencode_spec.aliases
    assert "chat.message" in opencode_spec.hook_events
    assert opencode_spec.plugin_builder is not None
    plugin = opencode_spec.build_plugin("trulens-client-hooks ingest opencode")
    assert "managed_by: trulens-client-hooks" in plugin
    assert "ingest opencode" in plugin
    assert "chat.message" in plugin


def test_client_registry_supports_direct_registration():
    clients.register_client(cursor_spec)
    clients.register_client(claude_spec)
    clients.register_client(opencode_spec)
    assert clients.get_client("cursor") is cursor_spec
    assert clients.get_client("claude") is claude_spec
    assert clients.get_client("opencode") is opencode_spec
    assert clients.get_client("open-code") is opencode_spec


def test_install_preserves_existing_hooks_is_idempotent_and_creates_backup(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps({
            "version": 2,
            "hooks": {"beforeSubmitPrompt": [{"command": "existing-hook"}]},
        })
    )
    spec = clients.ClientSpec(
        name="test-client",
        aliases=(),
        user_config_path=config_path,
        project_config_path=None,
        hook_events=("beforeSubmitPrompt",),
    )
    monkeypatch.setattr(clients, "get_client", lambda _: spec)

    assert cli.main(["install", "test-client"]) == 0
    first = json.loads(config_path.read_text())
    assert config_path.with_suffix(".json.trulens.bak").exists()
    assert first["version"] == 2
    assert first["hooks"]["beforeSubmitPrompt"][0]["command"] == "existing-hook"

    assert cli.main(["install", "test-client"]) == 0
    second = json.loads(config_path.read_text())
    assert second == first
    assert len(second["hooks"]["beforeSubmitPrompt"]) == 2


def test_uninstall_removes_only_trulens_hooks(tmp_path, monkeypatch):
    config_path = tmp_path / "settings.json"
    spec = clients.ClientSpec(
        name="test-client",
        aliases=(),
        user_config_path=config_path,
        project_config_path=None,
        hook_events=("Stop",),
    )
    monkeypatch.setattr(clients, "get_client", lambda _: spec)
    config_path.write_text(
        json.dumps({
            "hooks": {
                "Stop": [
                    {"command": "existing-hook"},
                    {"command": "trulens-client-hooks ingest test-client"},
                ]
            },
            "trulens": {"managed_by": "trulens-client-hooks"},
        })
    )

    assert cli.main(["uninstall", "test-client"]) == 0
    remaining = json.loads(config_path.read_text())
    assert remaining == {"hooks": {"Stop": [{"command": "existing-hook"}]}}


def test_plugin_parse_uses_canonical_opencode_name():
    clients.register_client(opencode_spec)
    event = parsers.parse(
        "opencode",
        {
            "session_id": "session-1",
            "hook_event_name": "session.idle",
            "model": "gpt-5",
        },
    )
    spans = tracing.TraceAssembler().assemble([event])
    agent = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.AGENT.value
    )
    assert event.client == "opencode"
    assert agent.attributes[SpanAttributes.WORKFLOW.AGENT_NAME] == "opencode"


def test_plugin_parse_uses_canonical_claude_name():
    clients.register_client(claude_spec)
    event = parsers.parse(
        "claude-code",
        {
            "session_id": "session-1",
            "hook_event_name": "Stop",
            "model": "claude-sonnet",
        },
    )
    spans = tracing.TraceAssembler().assemble([event])
    agent = next(
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.AGENT.value
    )
    assert event.client == "claude-code"
    assert agent.attributes[SpanAttributes.WORKFLOW.AGENT_NAME] == "claude-code"


def test_plugin_parse_rejects_missing_conversation_identity():
    clients.register_client(cursor_spec)
    with pytest.raises(ValueError, match="missing a conversation ID"):
        parsers.parse("cursor", {"hook_event_name": "stop"})


def test_opencode_parse_flattens_native_payloads():
    clients.register_client(opencode_spec)
    prompt = parsers.parse(
        "opencode",
        {
            "sessionID": "session-1",
            "messageID": "message-1",
            "hook_event_name": "chat.message",
            "model": {"providerID": "anthropic", "modelID": "opus"},
            "parts": [{"type": "text", "text": "hello"}],
        },
    )
    tool = parsers.parse(
        "opencode",
        {
            "sessionID": "session-1",
            "hook_event_name": "tool.execute.before",
            "tool": "bash",
            "callID": "call-1",
            "args": {"command": "pytest"},
        },
    )
    idle = parsers.parse(
        "opencode",
        {
            "event": {
                "type": "session.idle",
                "properties": {"sessionID": "session-1"},
            }
        },
    )

    assert prompt.client == "opencode"
    assert prompt.conversation_id == "session-1"
    assert prompt.turn_id == "message-1"
    assert prompt.prompt == "hello"
    assert prompt.model == "opus"
    assert tool.category == "tool"
    assert tool.phase == "start"
    assert tool.operation_id == "call-1"
    assert tool.tool_input == {"command": "pytest"}
    assert idle.terminal
    assert not idle.failed


def test_opencode_install_writes_managed_plugin_file(tmp_path, monkeypatch):
    plugin_path = tmp_path / "trulens-client-hooks.js"
    spec = clients.ClientSpec(
        name="opencode",
        aliases=(),
        user_config_path=plugin_path,
        project_config_path=None,
        hook_events=("chat.message",),
        plugin_builder=opencode_spec.plugin_builder,
    )
    monkeypatch.setattr(clients, "get_client", lambda _: spec)

    assert cli.main(["install", "opencode", "--dry-run"]) == 0
    assert cli.main(["install", "opencode"]) == 0
    contents = plugin_path.read_text()
    assert "managed_by: trulens-client-hooks" in contents
    assert not plugin_path.with_suffix(".js.trulens.bak").exists()
    assert cli.main(["install", "opencode"]) == 0
    assert plugin_path.with_suffix(".js.trulens.bak").exists()
    assert cli.main(["uninstall", "opencode"]) == 0
    assert not plugin_path.exists()
