"""Tests for the agent-hook run lifecycle.

Run management (creating a run and driving it to a terminal status via
``start_ingestion_query``) is a Snowflake AI Observability concept. OSS and
plain-OTLP destinations export spans but do not create or manage runs, so a
turn never carries a run status there. These tests pin down that split:
* OSS/SQLite sessions export spans without creating any run.
* Snowflake-capable destinations still create one run per conversation and
  complete each exported turn's ingestion.
* Run management can be disabled entirely via ``TRULENS_MANAGE_RUNS=false``.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest
from trulens.core import TruSession
from trulens.core.otel.client_hooks import journal
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import runs
from trulens.core.otel.client_hooks import service
from trulens.core.otel.client_hooks import tracing


def _cursor(event_name: str, conversation_id: str, **values):
    return {
        "conversation_id": conversation_id,
        "generation_id": values.pop("generation_id", "generation-1"),
        "hook_event_name": event_name,
        **values,
    }


@pytest.fixture
def hook_session(tmp_path: Path) -> TruSession:
    return TruSession(database_url=f"sqlite:///{tmp_path / 'hooks.sqlite'}")


def _journal_turn(
    event_journal: journal.EventJournal,
    conversation_id: str,
    generation_id: str = "generation-1",
) -> None:
    event_journal.append(
        parsers.parse_cursor(
            _cursor(
                "beforeSubmitPrompt",
                conversation_id,
                generation_id=generation_id,
                prompt="add a test",
            )
        )
    )
    event_journal.append(
        parsers.parse_cursor(
            _cursor(
                "stop",
                conversation_id,
                generation_id=generation_id,
                response="done",
            )
        )
    )


class _RecordingRunDao:
    """Recorded stand-in for a Snowflake RunDao (``snowflake_run_dao``)."""

    def __init__(self) -> None:
        self.ingestions = []

    def start_ingestion_query(
        self,
        object_name,
        object_version,
        object_type,
        run_name,
        input_records_count,
    ) -> None:
        self.ingestions.append((run_name, input_records_count))


class _FakeRun:
    def __init__(self, run_name, run_dao):
        self.run_name = run_name
        self.run_dao = run_dao
        self.object_name = "agent"
        self.object_type = "EXTERNAL AGENT"
        self.object_version = "v1"


class _SnowflakeCapableApp:
    """Mimics a TruApp after Snowflake ``augment_app`` fused a run DAO."""

    def __init__(self, snowflake_run_dao):
        self.snowflake_run_dao = snowflake_run_dao
        self.run_configs = []

    def add_run(self, run_config):
        self.run_configs.append(run_config)
        return _FakeRun(run_config.run_name, self.snowflake_run_dao)


def test_oss_session_exports_spans_without_runs(
    tmp_path: Path, hook_session: TruSession
):
    conversation_id = f"conversation-{uuid.uuid4().hex[:8]}"
    event_journal = journal.EventJournal(tmp_path / "journal")
    _journal_turn(event_journal, conversation_id)
    coordinator = runs.RunCoordinator(session=hook_session)
    hook_service = service.HookService(
        journal=event_journal,
        session=hook_session,
        coordinator=coordinator,
    )

    # Spans are exported and the turn is marked done, but no run is created for
    # an OSS destination.
    assert hook_service.flush()
    assert coordinator._runs == {}
    assert event_journal.pending_turns("cursor", conversation_id) == []


def test_oss_session_is_reported_unsupported(
    tmp_path: Path, hook_session: TruSession
):
    conversation_id = f"conversation-{uuid.uuid4().hex[:8]}"
    event_journal = journal.EventJournal(tmp_path / "journal")
    _journal_turn(event_journal, conversation_id)
    coordinator = runs.RunCoordinator(session=hook_session)
    service.HookService(
        journal=event_journal,
        session=hook_session,
        coordinator=coordinator,
    ).flush()

    # Runs are skipped and reported as unsupported exactly once.
    assert coordinator._unsupported_logged


def test_snowflake_capable_destination_creates_run_per_conversation():
    run_dao = _RecordingRunDao()
    app = _SnowflakeCapableApp(run_dao)
    coordinator = runs.RunCoordinator(session=object())
    key = ("cursor", "unknown")
    coordinator._apps[key] = app

    first = _identity_for("conversation-1", "generation-1")
    second = _identity_for("conversation-1", "generation-2")

    for identity in (first, second):
        run = coordinator.ensure_run(identity)
        assert run is not None
        assert coordinator.complete_turn(identity, run)

    # One run per conversation, reused across turns.
    assert len(app.run_configs) == 1
    assert app.run_configs[0].run_name == "conversation-1"
    # One completed invocation per turn, each contributing a single record.
    assert run_dao.ingestions == [
        ("conversation-1", 1),
        ("conversation-1", 1),
    ]


def test_non_snowflake_capable_destination_skips_runs():
    # An app exposing only a generic OSS run_dao (no snowflake_run_dao) must be
    # treated as unsupported, so no run is created.
    coordinator = runs.RunCoordinator(session=object())
    identity = _identity_for("conversation-1", "generation-1")

    assert coordinator.ensure_run(identity) is None
    assert coordinator.complete_turn(identity) is False


def test_run_lifecycle_can_be_disabled(
    tmp_path: Path,
    hook_session: TruSession,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRULENS_MANAGE_RUNS", "false")
    conversation_id = f"conversation-{uuid.uuid4().hex[:8]}"
    event_journal = journal.EventJournal(tmp_path / "journal")
    _journal_turn(event_journal, conversation_id)
    coordinator = runs.RunCoordinator(session=hook_session)
    hook_service = service.HookService(
        journal=event_journal,
        session=hook_session,
        coordinator=coordinator,
    )

    # Spans still export; there is simply no run to complete.
    assert hook_service.flush()
    assert coordinator._runs == {}
    assert event_journal.pending_turns("cursor", conversation_id) == []


def _identity_for(conversation_id: str, turn_id: str):
    events = [
        parsers.parse_cursor(
            _cursor(
                "beforeSubmitPrompt", conversation_id, generation_id=turn_id
            )
        ),
        parsers.parse_cursor(
            _cursor("stop", conversation_id, generation_id=turn_id)
        ),
    ]
    identity = tracing.TraceAssembler().identify(events)
    return tracing.TurnIdentity(
        client=identity.client,
        conversation_id=identity.conversation_id,
        turn_id=turn_id,
        record_id=f"cursor:{conversation_id}:{turn_id}",
        app_name=identity.app_name,
        app_version=identity.app_version,
        run_name=identity.run_name,
    )
