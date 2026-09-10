"""Tests for run status resolution from invocation metadata."""

from __future__ import annotations

from trulens.core.run import Run
from trulens.core.run import RunStatus


def _run_with_invocation(completion_status) -> Run:
    """Build a Run carrying a single invocation with ``completion_status``."""

    run = Run.model_construct(
        run_dao=None,
        app=None,
        tru_session=None,
        object_name="agent",
        object_type="EXTERNAL AGENT",
        object_version="v1",
        run_name="conversation-1",
    )
    run.run_metadata = Run.RunMetadata(
        invocations={
            "invocation_1": Run.InvocationMetadata(
                id="invocation_1",
                input_records_count=1,
                start_time_ms=1,
                end_time_ms=2,
                completion_status=completion_status,
            )
        }
    )
    return run


def test_completed_invocation_resolves_to_terminal_status():
    run = _run_with_invocation(
        Run.CompletionStatus(
            status=Run.CompletionStatusStatus.COMPLETED, record_count=1
        )
    )

    assert (
        run._compute_latest_invocation_status(run)
        == RunStatus.INVOCATION_COMPLETED
    )


def test_started_invocation_resolves_to_in_progress():
    run = _run_with_invocation(
        Run.CompletionStatus(status=Run.CompletionStatusStatus.STARTED)
    )

    assert (
        run._compute_latest_invocation_status(run)
        == RunStatus.INVOCATION_IN_PROGRESS
    )


def test_invocation_without_completion_status_still_resolves():
    """An invocation with no completion status must not yield ``None``.

    Returning ``None`` makes the status unresolvable for callers and renderers,
    which surfaces as a run that spins indefinitely with no explanation.
    """

    run = _run_with_invocation(None)

    status = run._compute_latest_invocation_status(run)

    assert status is not None
    assert status == RunStatus.INVOCATION_IN_PROGRESS
    assert isinstance(status, RunStatus)
