"""Contract tests for thin coding-agent client plugins."""

import json

import pytest
from trulens.apps.claude import client_spec as claude_spec
from trulens.apps.cursor import client_spec as cursor_spec
from trulens.core.otel.client_hooks import cli
from trulens.core.otel.client_hooks import clients
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import tracing
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes


def test_cursor_plugin_is_declarative():
    assert cursor_spec.name == "cursor"
    assert "afterFileEdit" in cursor_spec.hook_events
    assert cursor_spec.field_aliases.conversation == ("conversation_id",)


def test_claude_plugin_is_declarative():
    assert claude_spec.name == "claude-code"
    assert "claude" in claude_spec.aliases
    assert claude_spec.field_aliases.conversation == ("session_id",)


def test_client_registry_supports_direct_registration():
    clients.register_client(cursor_spec)
    clients.register_client(claude_spec)
    assert clients.get_client("cursor") is cursor_spec
    assert clients.get_client("claude") is claude_spec


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


def test_plugin_parse_uses_canonical_claude_name_and_anthropic_system():
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
    assert agent.attributes[GenAIAttributes.SYSTEM.NAME] == "anthropic"


def test_plugin_parse_rejects_missing_conversation_identity():
    clients.register_client(cursor_spec)
    with pytest.raises(ValueError, match="missing a conversation ID"):
        parsers.parse("cursor", {"hook_event_name": "stop"})
