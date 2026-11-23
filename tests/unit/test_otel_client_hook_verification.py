"""Tests for test, build and exit-code signals on coding-agent hook spans."""

from __future__ import annotations

import json

import pytest
from trulens.core.otel.client_hooks import classify
from trulens.core.otel.client_hooks import models
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import privacy
from trulens.core.otel.client_hooks import tracing
from trulens.otel.semconv.trace import SpanAttributes

_CODING_AGENT = SpanAttributes.CODING_AGENT


def _claude(event_name: str, **values):
    return {"session_id": "session-1", "hook_event_name": event_name, **values}


def _cursor(event_name: str, **values):
    return {
        "conversation_id": "conversation-1",
        "generation_id": "generation-1",
        "hook_event_name": event_name,
        **values,
    }


def _opencode(event_name: str, **values):
    return {
        "session_id": "session-1",
        "message_id": "message-1",
        "hook_event_name": event_name,
        **values,
    }


def _failing_pytest_turn(client: str):
    """Native payloads for one shell call running `pytest -q tests/` that fails."""

    command = "pytest -q tests/"
    if client == "claude":
        return [
            _claude(
                "PreToolUse",
                tool_name="Bash",
                tool_use_id="bash-1",
                tool_input={"command": command},
            ),
            _claude(
                "PostToolUseFailure",
                tool_name="Bash",
                tool_use_id="bash-1",
                tool_input={"command": command},
                error="Exit code 1\n1 failed, 3 passed",
            ),
        ]
    if client == "cursor":
        return [
            _cursor(
                "beforeShellExecution",
                command=command,
                operation_id="shell-1",
            ),
            _cursor(
                "afterShellExecution",
                command=command,
                output="1 failed, 3 passed",
                exit_code=1,
                operation_id="shell-1",
            ),
        ]
    return [
        _opencode(
            "tool.execute.before",
            tool_name="bash",
            tool_call_id="call-1",
            tool_input={"command": command},
        ),
        _opencode(
            "tool.execute.after",
            tool_name="bash",
            tool_call_id="call-1",
            tool_input={"command": command},
            tool_output="1 failed, 3 passed",
            exit_code=1,
        ),
    ]


def _test_file_edit(client: str, path: str):
    """One native edit payload for each client."""

    if client == "claude":
        return _claude(
            "PostToolUse",
            tool_name="Edit",
            tool_use_id="edit-1",
            tool_input={
                "file_path": path,
                "old_string": "a",
                "new_string": "b",
            },
        )
    if client == "cursor":
        return _cursor(
            "afterFileEdit",
            file_path=path,
            edits=[{"old_string": "a", "new_string": "b"}],
        )
    return _opencode("file.edited", file_path=path)


def _tool_spans(client: str, payloads, policy=None):
    policy = policy or privacy.CapturePolicy()
    events = [
        policy.apply(parsers.parse(client, payload)) for payload in payloads
    ]
    spans = tracing.TraceAssembler().assemble(events)
    return [
        span
        for span in spans
        if span.attributes[SpanAttributes.SPAN_TYPE]
        == SpanAttributes.SpanType.TOOL.value
    ]


@pytest.mark.parametrize(
    ("command", "kind"),
    [
        ("pytest -q tests/", "test"),
        ("python -m pytest tests/unit", "test"),
        ("uv run pytest -x", "test"),
        ("PYTHONPATH=src poetry run pytest", "test"),
        ("cd web && npm test", "test"),
        ("npm run test:e2e", "test"),
        ("make test-unit", "test"),
        ("cargo test --workspace", "test"),
        ("go test ./...", "test"),
        ("git add -A && pytest", "test"),
        ("tsc --noEmit", "build"),
        ("npm run build", "build"),
        ("cargo build --release", "build"),
        ("make", "build"),
        ("ruff check .", "lint"),
        ("npx eslint src", "lint"),
        ("make lint", "lint"),
        ("git status", "vcs"),
        ('git commit -m "fix pytest flake"', "vcs"),
        ("ls tests/", "other"),
        ("cat pytest.ini", "other"),
        ("echo 'npm test'", "other"),
    ],
)
def test_classify_command_kinds(command, kind):
    assert classify.classify_command(command, environ={}) == kind


def test_classify_command_without_command_is_none():
    assert classify.classify_command(None, environ={}) is None
    assert classify.classify_command("  ", environ={}) is None


def test_command_kind_patterns_extend_defaults():
    environ = {
        classify.COMMAND_KIND_PATTERNS_ENV: json.dumps({
            "test": ["./scripts/run-e2e"],
            "build": "bazel run //:gen",
        })
    }

    assert classify.classify_command("./scripts/run-e2e --ci", environ) == (
        "test"
    )
    assert classify.classify_command("bazel run //:gen", environ) == "build"
    assert classify.classify_command("pytest", environ) == "test"


def test_invalid_command_kind_patterns_are_ignored():
    environ = {classify.COMMAND_KIND_PATTERNS_ENV: "not json"}

    assert classify.classify_command("pytest", environ) == "test"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_x.py", True),
        ("/repo/pkg/tests/helpers.py", True),
        ("src/test_app.py", True),
        ("src/app_test.py", True),
        ("web/src/__tests__/button.tsx", True),
        ("web/src/button.spec.ts", True),
        ("web/src/button.test.tsx", True),
        ("conftest.py", True),
        ("./tests/data/fixture.json", True),
        ("src/app.py", False),
        ("src/contest.py", False),
        ("docs/testing.md", False),
    ],
)
def test_is_test_file_default_globs(path, expected):
    assert classify.is_test_file(path, environ={}) is expected


def test_test_file_globs_override_defaults():
    environ = {classify.TEST_FILE_GLOBS_ENV: "spec/**, *_spec.rb"}

    assert classify.is_test_file("spec/models/user_spec.rb", environ)
    assert classify.is_test_file("lib/user_spec.rb", environ)
    assert not classify.is_test_file("tests/test_x.py", environ)


def test_is_test_file_without_path_is_none():
    assert classify.is_test_file(None, environ={}) is None


@pytest.mark.parametrize("client", ["claude", "cursor", "opencode"])
def test_failing_test_run_is_recorded_without_command(client):
    tool_spans = _tool_spans(client, _failing_pytest_turn(client))

    assert len(tool_spans) == 1
    attributes = tool_spans[0].attributes
    assert attributes[_CODING_AGENT.EXIT_CODE] == 1
    assert attributes[_CODING_AGENT.COMMAND_KIND] == "test"
    assert _CODING_AGENT.COMMAND not in attributes
    assert "pytest" not in json.dumps(dict(attributes))


@pytest.mark.parametrize("client", ["claude", "cursor", "opencode"])
def test_command_is_recorded_with_tool_payload_capture(client):
    tool_spans = _tool_spans(
        client,
        _failing_pytest_turn(client),
        policy=privacy.CapturePolicy(capture_tool_payloads=True),
    )

    assert tool_spans[0].attributes[_CODING_AGENT.COMMAND] == (
        "pytest -q tests/"
    )


@pytest.mark.parametrize("client", ["claude", "cursor", "opencode"])
def test_test_file_edit_is_flagged_without_path(client):
    tool_spans = _tool_spans(
        client, [_test_file_edit(client, "tests/test_x.py")]
    )

    assert len(tool_spans) == 1
    attributes = tool_spans[0].attributes
    assert attributes[_CODING_AGENT.IS_TEST_FILE] is True
    assert "test_x.py" not in json.dumps(dict(attributes))


@pytest.mark.parametrize("client", ["claude", "cursor", "opencode"])
def test_source_file_edit_is_not_a_test_file(client):
    tool_spans = _tool_spans(client, [_test_file_edit(client, "src/app.py")])

    assert tool_spans[0].attributes[_CODING_AGENT.IS_TEST_FILE] is False


def test_claude_successful_bash_records_exit_code_zero():
    event = parsers.parse_claude(
        _claude(
            "PostToolUse",
            tool_name="Bash",
            tool_use_id="bash-1",
            tool_input={"command": "pytest"},
            tool_response={"stdout": "4 passed", "stderr": ""},
        )
    )

    assert event.exit_code == 0
    assert event.command_kind == "test"


def test_claude_failure_without_code_in_error_records_exit_code_one():
    event = parsers.parse_claude(
        _claude(
            "PostToolUseFailure",
            tool_name="Bash",
            tool_use_id="bash-1",
            tool_input={"command": "pytest"},
            error="Command failed",
        )
    )

    assert event.exit_code == 1


def test_opencode_reads_exit_code_from_structured_output():
    event = parsers.parse_opencode(
        _opencode(
            "tool.execute.after",
            tool_name="shell",
            tool_call_id="call-1",
            tool_input={"command": "go test ./..."},
            tool_output={"output": "FAIL", "exitCode": 2},
        )
    )

    assert event.exit_code == 2
    assert event.command_kind == "test"


def test_exit_code_no_longer_lands_in_metadata():
    event = parsers.parse_cursor(
        _cursor("afterShellExecution", command="ls", exit_code=0)
    )

    assert event.exit_code == 0
    assert "exit_code" not in event.metadata


def test_shell_start_event_has_no_exit_code():
    event = parsers.parse_cursor(
        _cursor("beforeShellExecution", command="pytest")
    )

    assert event.exit_code is None
    assert event.command_kind == "test"


def test_non_shell_tool_is_not_classified():
    event = parsers.parse_claude(
        _claude(
            "PostToolUse",
            tool_name="Read",
            tool_use_id="read-1",
            tool_input={"file_path": "tests/test_x.py"},
        )
    )

    assert event.exit_code is None
    assert event.command_kind is None
    assert event.is_test_file is None


def test_capture_policy_keeps_labels_and_drops_command_by_default():
    event = parsers.parse_claude(
        _claude(
            "PostToolUseFailure",
            tool_name="Bash",
            tool_use_id="bash-1",
            tool_input={"command": "pytest"},
            error="Exit code 3",
        )
    )

    captured = privacy.CapturePolicy().apply(event)

    assert captured.command is None
    assert captured.tool_input is None
    assert captured.exit_code == 3
    assert captured.command_kind == "test"


def test_journaled_event_without_new_fields_still_loads():
    event = parsers.parse_cursor(_cursor("stop", status="completed"))
    stored = event.to_dict()
    for key in ("command", "exit_code", "command_kind", "is_test_file"):
        stored.pop(key)

    restored = models.HookEvent.from_dict(stored)

    assert restored.exit_code is None
    assert restored.is_test_file is None
