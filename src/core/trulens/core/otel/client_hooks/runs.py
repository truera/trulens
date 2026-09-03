"""AI Observability run lifecycle for coding-agent hook exports.

Exporting spans is not sufficient to make a turn observable. A run's displayed
status is derived from its run metadata, not from the presence of spans, and the
only writer that creates and completes invocation metadata is the ingestion
query started through
`RunDaoBase.start_ingestion_query`.
Without it a run stays non-terminal forever and renders as perpetually
in-progress.

Each conversation maps to one run, and each exported turn contributes one
completed invocation to that run. The run therefore reaches a terminal status
after the first turn and stays terminal as later turns arrive, because run
status resolves against the most recent invocation.

Every operation here distinguishes an unsupported destination from a genuine
failure. Destinations with no run concept report "not applicable" and are
skipped quietly; real failures raise so the caller can apply the journal's
existing retry and backoff, because a turn whose ingestion never started is not
actually finished.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from trulens.core import session as core_session
from trulens.core.enums import Mode
from trulens.core.otel.client_hooks import tracing
from trulens.core.run import RunConfig

logger = logging.getLogger(__name__)

_DATASET_NAME = "client_hooks"
"""Dataset name recorded on hook-created runs.

Hook turns are ingested from the local journal rather than a user table, so this
is descriptive metadata only.
"""

_DATASET_SPEC = {
    "input_id": "input_id",
    "record_root.input": "input",
    "record_root.output": "output",
}
"""Column spec recorded on hook-created runs.

``run.start()`` is never called for hook runs -- spans are assembled from
journalled events and exported directly -- so this spec is metadata describing
the shape of the emitted records rather than a mapping that gets read back.
"""


class _HookTurnSource:
    """Placeholder app object for hook-created runs.

    Runs require an owning app, but hook turns are recorded from native client
    events rather than by invoking Python code, so there is no real app to wrap.
    """


def runs_enabled() -> bool:
    """Return whether hook exports should manage run lifecycle.

    Set ``TRULENS_MANAGE_RUNS=false`` to export spans without creating runs or
    starting ingestion. Turns then remain non-terminal in the UI, so this is
    intended for debugging the span path in isolation.
    """

    value = os.environ.get("TRULENS_MANAGE_RUNS", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_snowflake_run_capable(app: Any) -> bool:
    """Return whether ``app`` owns a Snowflake AI Observability run store.

    Run lifecycle (creating a run and driving it to a terminal status via
    ``start_ingestion_query``) is a Snowflake AI Observability concept. The
    Snowflake connector fuses a Snowflake ``RunDao`` onto the app as
    ``snowflake_run_dao`` during ``augment_app``; OSS connectors only provide a
    generic ``DefaultRunDao`` under ``run_dao``. Checking for the Snowflake-only
    attribute (rather than any ``run_dao``) keeps run management scoped to
    Snowflake destinations without importing the Snowflake connector into
    trulens-core.
    """

    return getattr(app, "snowflake_run_dao", None) is not None


class RunCoordinator:
    """Create and complete runs for exported coding-agent turns.

    Apps and runs are cached per process because a hook export drains many turns
    from the same conversation, and each ``add_run`` or app construction is a
    round trip to the backend.
    """

    def __init__(
        self,
        *,
        session: Optional[core_session.TruSession] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.session = session
        self.enabled = runs_enabled() if enabled is None else enabled
        self._lock = threading.Lock()
        self._apps: Dict[Tuple[str, str], Any] = {}
        self._runs: Dict[Tuple[str, str, str], Any] = {}
        self._unsupported_logged = False

    def _unsupported(self, reason: str) -> None:
        """Record that the destination has no run concept, warning once."""

        if not self._unsupported_logged:
            self._unsupported_logged = True
            logger.warning(
                "%s Exporting spans without run lifecycle; turns will not reach "
                "a terminal run status.",
                reason,
            )

    def _get_app(self, identity: tracing.TurnIdentity) -> Optional[Any]:
        """Build or return the cached app owning ``identity``'s run.

        Returns ``None`` when the destination has no run concept, and raises if
        an app that should exist could not be constructed.
        """

        key = (identity.app_name, identity.app_version)
        if key in self._apps:
            return self._apps[key]

        from trulens.apps.app import TruApp

        session = self.session
        connector = getattr(session, "connector", None) if session else None
        if connector is None:
            # Destinations reached without a connector -- plain OTLP, for
            # example -- can still receive spans, but there is no run store to
            # create a run in or an invocation to complete.
            self._unsupported("Destination has no connector.")
            return None
        # Snowflake connectors register an EXTERNAL AGENT and attach a run DAO
        # during construction; object_type is deliberately left unset so each
        # connector applies its own default.
        app = TruApp(
            _HookTurnSource(),
            app_name=identity.app_name,
            app_version=identity.app_version,
            connector=connector,
            start_evaluator=False,
        )
        if not _is_snowflake_run_capable(app):
            self._unsupported(
                "Destination does not support Snowflake AI Observability runs."
            )
            return None
        self._apps[key] = app
        return app

    def ensure_run(self, identity: tracing.TurnIdentity) -> Optional[Any]:
        """Ensure the run for ``identity`` exists before its spans are exported.

        The run must exist first because exported spans carry its name.

        Returns the run, or ``None`` when run management is disabled or the
        destination has no run concept. Raises if the run could not be created,
        so the caller retries the turn rather than exporting spans that would
        never reach a terminal status.
        """

        if not self.enabled:
            return None
        key = (identity.app_name, identity.app_version, identity.run_name)
        with self._lock:
            if key in self._runs:
                return self._runs[key]
            app = self._get_app(identity)
            if app is None:
                return None
            # add_run is idempotent: it returns the existing run when one
            # already exists, so repeated turns and retries are safe.
            run = app.add_run(
                RunConfig(
                    run_name=identity.run_name,
                    dataset_name=_DATASET_NAME,
                    source_type="DATAFRAME",
                    mode=Mode.LOG_INGESTION,
                    dataset_spec=dict(_DATASET_SPEC),
                    description=(
                        f"{identity.client} conversation "
                        f"{identity.conversation_id}"
                    ),
                    label=identity.client,
                )
            )
            if run is None:
                # The destination supports runs, so a missing run here is an
                # unexpected state rather than a quiet opt-out. Failing loudly
                # keeps the turn retryable instead of exporting spans that would
                # never reach a terminal status.
                raise RuntimeError(
                    f"Run {identity.run_name!r} could not be created for "
                    f"{identity.app_name!r}."
                )
            self._runs[key] = run
            return run

    def complete_turn(
        self, identity: tracing.TurnIdentity, run: Optional[Any] = None
    ) -> bool:
        """Start ingestion for one exported turn.

        This is the call that drives the turn's invocation to a terminal status.
        It runs after a successful span export so that the ingestion window does
        not open before the spans it waits for have been sent.

        Returns whether ingestion was started; ``False`` means run management is
        disabled or unsupported. Raises if ingestion could not be started.
        """

        if not self.enabled:
            return False
        run = run or self._runs.get((
            identity.app_name,
            identity.app_version,
            identity.run_name,
        ))
        if run is None:
            return False
        run.run_dao.start_ingestion_query(
            object_name=run.object_name,
            object_type=run.object_type,
            object_version=run.object_version,
            run_name=identity.run_name,
            input_records_count=identity.input_records_count,
        )
        return True
