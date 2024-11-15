from dataclasses import dataclass
from typing import Iterable, Optional

from trulens.core.database import base as core_db
from trulens.core.schema import types as types_schema
from trulens.experimental.otel_tracing.core.trace import sem as core_sem


@dataclass
class SpanIndex:
    """Index of a span in the database.

    A span can be indexed either by index alone or a combination of span_id and
    trace_id.
    """

    index: Optional[int] = None
    span_id: Optional[types_schema.SpanID.SQL_TYPE] = None
    trace_id: Optional[types_schema.TraceID.SQL_TYPE] = None


class _DB(core_db.DB):
    # EXPERIMENTAL(otel_tracing): Adds span API to the core DB API.

    T = core_db.DB.T
    Q = core_db.DB.Q
    W = core_db.DB.W

    def insert_span(self, span: core_sem.TypedSpan) -> None:
        """Insert a span into the database."""
        raise NotImplementedError

    def batch_insert_span(self, spans: Iterable[core_sem.TypedSpan]) -> None:
        """Insert a batch of spans into the database."""
        raise NotImplementedError

    def delete_span(self, index: SpanIndex) -> None:
        """Delete a span from the database."""
        raise NotImplementedError

    def delete_spans(
        self,
        where: W,
        page: Optional[core_db.PageSelect] = None,
    ) -> None:
        """Delete spans from the database."""
        raise NotImplementedError

    def get_span(self, index: SpanIndex) -> Optional[core_sem.TypedSpan]:
        """Get a span from the database."""
        raise NotImplementedError

    def get_spans(
        self,
        where: W,
        page: Optional[core_db.PageSelect] = None,
    ) -> Iterable[core_sem.TypedSpan]:
        """Select spans from the database."""
        raise NotImplementedError

    def get_trace_record_ids(
        self, where: W, page: Optional[core_db.PageSelect] = None
    ) -> Iterable[str]:
        """Get the trace record ids matching the given query/page."""

        raise NotImplementedError

    def get_trace_record(self, record_id: str) -> Iterable[core_sem.TypedSpan]:
        """Select spans from the database that belong to the given record."""

        raise NotImplementedError
