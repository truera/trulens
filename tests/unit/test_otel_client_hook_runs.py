"""End-to-end test that exported hook turns reach a terminal run status.

Span export alone does not make a turn observable: a run's status is derived
from its invocation metadata, so a turn whose ingestion never started renders as
perpetually in-progress. These tests assert on run status rather than on span
counts, which is the signal that distinguishes a working export from a spinner.
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
from trulens.core.run import RunStatus


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


def test_flushed_turn_reaches_terminal_run_status(
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

    assert hook_service.flush()

    run = coordinator._runs[("cursor", "unknown", conversation_id)]
    assert run.get_status() == RunStatus.INVOCATION_COMPLETED


def test_second_turn_keeps_run_terminal(
    tmp_path: Path, hook_session: TruSession
):
    conversation_id = f"conversation-{uuid.uuid4().hex[:8]}"
    event_journal = journal.EventJournal(tmp_path / "journal")
    coordinator = runs.RunCoordinator(session=hook_session)
    hook_service = service.HookService(
        journal=event_journal,
        session=hook_session,
        coordinator=coordinator,
    )

    _journal_turn(event_journal, conversation_id, "generation-1")
    assert hook_service.flush()
    _journal_turn(event_journal, conversation_id, "generation-2")
    assert hook_service.flush()

    # One run per conversation, still terminal after a later turn arrives.
    assert list(coordinator._runs) == [("cursor", "unknown", conversation_id)]
    run = coordinator._runs[("cursor", "unknown", conversation_id)]
    assert run.get_status() == RunStatus.INVOCATION_COMPLETED


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
