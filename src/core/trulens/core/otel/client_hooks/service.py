"""End-to-end hook event ingestion service."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from trulens.core import session as core_session
from trulens.core.otel.client_hooks import exporting
from trulens.core.otel.client_hooks import journal as journal_module
from trulens.core.otel.client_hooks import parsers
from trulens.core.otel.client_hooks import privacy
from trulens.core.otel.client_hooks import tracing


class HookService:
    """Normalize, sanitize, journal, assemble, and export hook events."""

    def __init__(
        self,
        *,
        journal: Optional[journal_module.EventJournal] = None,
        capture_policy: Optional[privacy.CapturePolicy] = None,
        assembler: Optional[tracing.TraceAssembler] = None,
        session: Optional[core_session.TruSession] = None,
    ) -> None:
        self.journal = journal or journal_module.EventJournal()
        self.capture_policy = (
            capture_policy or privacy.CapturePolicy.from_environment()
        )
        self.assembler = assembler or tracing.TraceAssembler(
            app_name=os.environ.get("TRULENS_HOOKS_APP_NAME"),
            app_version=os.environ.get("TRULENS_HOOKS_APP_VERSION")
            or os.environ.get("TRULENS_HOOKS_CLIENT_VERSION"),
            run_name=os.environ.get("TRULENS_HOOKS_RUN_NAME"),
        )
        self.session = session

    def ingest(self, client: str, payload: Mapping[str, Any]) -> bool:
        """Process one hook payload and export any newly complete turns."""

        event = self.capture_policy.apply(parsers.parse(client, payload))
        self.journal.append(event)
        return self.flush()

    def flush(self) -> bool:
        """Export all complete, retryable, or stale turns in the journal."""

        success = True
        for client, conversation_id in self.journal.conversations():
            for turn_id in self.journal.claim_pending_turns(
                client,
                conversation_id,
                stale_after=tracing.stale_after_from_environment(),
            ):
                turn = self.journal.get_turn(client, conversation_id, turn_id)
                stale = not any(item.terminal for item in turn)
                spans = self.assembler.assemble(turn, stale=stale)
                if exporting.export_spans(spans, session=self.session):
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
