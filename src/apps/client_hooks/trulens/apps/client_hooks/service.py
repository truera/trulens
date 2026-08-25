"""End-to-end hook event ingestion service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from trulens.apps.client_hooks import exporting
from trulens.apps.client_hooks import journal as journal_module
from trulens.apps.client_hooks import parsers
from trulens.apps.client_hooks import privacy
from trulens.apps.client_hooks import tracing
from trulens.core import session as core_session


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
        self.assembler = assembler or tracing.TraceAssembler()
        self.session = session

    def ingest(self, client: str, payload: Mapping[str, Any]) -> bool:
        """Process one hook payload and export any newly complete turns."""

        event = self.capture_policy.apply(parsers.parse(client, payload))
        self.journal.append(event)
        success = True
        for turn_id in self.journal.claim_pending_turns(
            event.client,
            event.conversation_id,
            stale_after=tracing.stale_after_from_environment(),
        ):
            turn = self.journal.get_turn(
                event.client, event.conversation_id, turn_id
            )
            stale = not any(item.terminal for item in turn)
            spans = self.assembler.assemble(turn, stale=stale)
            if exporting.export_spans(spans, session=self.session):
                self.journal.mark_exported(
                    event.client, event.conversation_id, turn_id
                )
            else:
                self.journal.release_claim(
                    event.client,
                    event.conversation_id,
                    turn_id,
                    failed=True,
                )
                success = False
        return success
