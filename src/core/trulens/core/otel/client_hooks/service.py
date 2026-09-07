"""End-to-end hook event ingestion service."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Tuple

from trulens.core import session as core_session
from trulens.core.otel.client_hooks import exporting
from trulens.core.otel.client_hooks import journal as journal_module
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import privacy
from trulens.core.otel.client_hooks import runs
from trulens.core.otel.client_hooks import tracing

logger = logging.getLogger(__name__)


class HookService:
    """Normalize, sanitize, journal, assemble, and export hook events."""

    def __init__(
        self,
        *,
        journal: Optional[journal_module.EventJournal] = None,
        capture_policy: Optional[privacy.CapturePolicy] = None,
        assembler: Optional[tracing.TraceAssembler] = None,
        session: Optional[core_session.TruSession] = None,
        coordinator: Optional[runs.RunCoordinator] = None,
    ) -> None:
        self.journal = journal or journal_module.EventJournal()
        self.capture_policy = (
            capture_policy or privacy.CapturePolicy.from_environment()
        )
        self.assembler = assembler or tracing.TraceAssembler(
            app_name=os.environ.get("TRULENS_APP_NAME"),
            app_version=os.environ.get("TRULENS_APP_VERSION"),
            run_name=os.environ.get("TRULENS_RUN_NAME"),
        )
        self.stale_after = tracing.stale_after_from_environment()
        self.session = session
        self.coordinator = coordinator or runs.RunCoordinator()

    def _resolve_session(self) -> core_session.TruSession:
        """Create the destination session once and reuse it across turns.

        A flush drains every pending turn, so building a session per turn would
        reconnect to the destination repeatedly.
        """

        if self.session is None:
            self.session = exporting.create_session()
        if self.coordinator.session is None:
            self.coordinator.session = self.session
        return self.session

    def ingest(
        self, client: str, payload: Mapping[str, Any]
    ) -> Tuple[str, bool]:
        """Normalize and durably journal one hook payload without exporting."""

        event = self.capture_policy.apply(parsers.parse(client, payload))
        return self.journal.append(event)

    def flush(self) -> bool:
        """Export all complete, retryable, or stale turns in the journal.

        Each turn's run is created before its spans are exported, because the
        spans carry the run name, and its ingestion is started afterwards, which
        is what drives the turn's invocation to a terminal status. A turn is only
        marked exported once both have succeeded; otherwise it is released for
        retry, since spans without ingestion would leave the run in-progress
        forever.
        """

        success = True
        for client, conversation_id in self.journal.conversations():
            for turn_id in self.journal.claim_pending_turns(
                client,
                conversation_id,
                stale_after=self.stale_after,
                lease_for=journal_module.export_lease_from_environment(),
            ):
                turn = self.journal.get_turn(client, conversation_id, turn_id)
                stale = not any(item.terminal for item in turn)
                identity = self.assembler.identify(turn)
                spans = self.assembler.assemble(turn, stale=stale)
                try:
                    session = self._resolve_session()
                    run = (
                        None
                        if identity is None
                        else self.coordinator.ensure_run(identity)
                    )
                    exported = exporting.export_spans(spans, session=session)
                    if exported and identity is not None:
                        try:
                            self.coordinator.complete_turn(identity, run)
                        except Exception:
                            logger.warning(
                                "Spans exported, but run lifecycle completion "
                                "failed. The trace is available, but its run "
                                "may remain non-terminal.",
                                exc_info=True,
                            )
                except Exception:
                    self.journal.release_claim(
                        client,
                        conversation_id,
                        turn_id,
                        failed=True,
                    )
                    success = False
                    continue
                if exported:
                    self.journal.mark_exported(client, conversation_id, turn_id)
                else:
                    self.journal.release_claim(
                        client,
                        conversation_id,
                        turn_id,
                        failed=True,
                    )
                    success = False
        return success
