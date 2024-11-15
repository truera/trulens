from typing import Iterable, List, Optional

import sqlalchemy as sa
from trulens.core.database import base as core_db
from trulens.core.database import sqlalchemy as core_sqlalchemy
from trulens.core.schema import types as types_schema
from trulens.experimental.otel_tracing.core.database import base as otel_core_db
from trulens.experimental.otel_tracing.core.database import orm as otel_core_orm
from trulens.experimental.otel_tracing.core.trace import sem as core_sem


class _SQLAlchemyDB(core_sqlalchemy.SQLAlchemyDB):
    """EXPERIMENTAL(otel_tracing): Adds span API to the SQLAlchemy DB API."""

    T = core_sqlalchemy.SQLAlchemyDB.T
    """Table type."""

    Q = core_sqlalchemy.SQLAlchemyDB.Q
    """Query type."""

    W = core_sqlalchemy.SQLAlchemyDB.W
    """Where clause type."""

    @property
    def Span(self) -> T:
        """Span query."""
        if self.orm is None:
            raise RuntimeError("ORM not set. Cannot refer to Span table.")

        if not hasattr(self.orm, "Span"):
            raise RuntimeError(
                "ORM does not have Span table. It might be an unpatched ORM class without otel_tracing support."
            )

        return self.orm.Span

    def _where_span_index(
        self,
        query: Q,
        index: otel_core_db.SpanIndex,
    ) -> Q:
        """Adds where clauses to the given Span query based on the given index."""

        self.orm: (
            otel_core_orm.SpanORM
        )  # assume self.orm was patched with otel_tracing additions

        if index.index is None and (
            index.span_id is None or index.trace_id is None
        ):
            raise ValueError(
                "Span index must have either index or (span_id and trace_id)."
            )

        if index.index is not None:
            query = query.where(self.Span.index == index.index)

        if index.span_id is not None:
            query = query.where(self.Span.span_id == index.span_id)

        if index.trace_id is not None:
            query = query.where(self.Span.trace_id == index.trace_id)

        return query

    def _where_page(self, query: Q, page: core_db.PageSelect) -> Q:
        """Adds paging clauses to the given query based on the given page select."""

        if page.limit is not None:
            query = query.limit(page.limit)

        if page.offset is not None:
            query = query.offset(page.offset)

        if page.shuffle:
            query = query.order_by(sa.func.random())

        if page.after_index is not None:
            query = query.where(self.Span.index > page.after_index)

        if page.before_index is not None:
            query = query.where(self.Span.index < page.before_index)

        if page.after_created_timestamp:
            query = query.where(
                self.orm.Span.created_timestamp > page.after_created_timestamp
            )

        if page.before_created_timestamp:
            query = query.where(
                self.Span.created_timestamp < page.before_created_timestamp
            )

        if page.after_updated_timestamp:
            query = query.where(
                self.Span.updated_timestamp > page.after_updated_timestamp
            )

        if page.before_updated_timestamp:
            query = query.where(
                self.Span.updated_timestamp < page.before_updated_timestamp
            )

        return query

    def insert_span(self, span: core_sem.TypedSpan) -> otel_core_db.SpanIndex:
        """Insert a span into the database."""

        if (
            span.context.span_id == types_schema.SpanID.INVALID_OTEL
            or span.context.trace_id == types_schema.TraceID.INVALID_OTEL
        ):
            raise ValueError(f"Invalid span context: {span.context}")

        with self.session.begin() as session:
            orm_object = self.Span.parse(span)
            session.merge(orm_object)

            return otel_core_db.SpanIndex(
                index=orm_object.index,
                span_id=orm_object.span_id,
                trace_id=orm_object.trace_id,
            )

    def insert_spans(
        self, spans: Iterable[core_sem.TypedSpan]
    ) -> List[otel_core_db.SpanIndex]:
        """Insert a batch of spans into the database."""

        for span in spans:
            if (
                span.context.span_id == types_schema.SpanID.INVALID_OTEL
                or span.context.trace_id == types_schema.TraceID.INVALID_OTEL
            ):
                raise ValueError(f"Invalid span context: {span.context}")

        with self.session.begin() as session:
            orm_objects = [self.Span.parse(span) for span in spans]
            session.add_all(orm_objects)

            return [
                otel_core_db.SpanIndex(
                    index=orm_object.index,
                    span_id=orm_object.span_id,
                    trace_id=orm_object.trace_id,
                )
                for orm_object in orm_objects
            ]

    def delete_span(self, index: otel_core_db.SpanIndex) -> None:
        """Delete a span from the database."""

        with self.session.begin() as session:
            query = self._where_span_index(
                query=session.query(self.Span), index=index
            ).first()
            session.delete(query)

    def delete_spans(
        self,
        where: Optional[W] = None,
        page: Optional[core_db.PageSelect] = None,
    ) -> None:
        """Delete spans from the database."""

        with self.session.begin() as session:
            query = session.query(self.Span)
            if where is not None:
                query = query.where(where)

            if page is not None:
                query = self._where_page(query=query, page=page)

            session.delete(query)

    def get_span(
        self, index: otel_core_db.SpanIndex
    ) -> Optional[core_sem.TypedSpan]:
        """Get a span from the database."""

        with self.session.begin() as session:
            query = self._where_span_index(
                query=session.query(self.Span), index=index
            ).first()
            orm_object = query.one_or_none()

            if orm_object is None:
                return None

            return orm_object.write()

    def get_spans(
        self,
        where: Optional[W] = None,
        page: Optional[core_db.PageSelect] = None,
    ) -> Iterable[core_sem.TypedSpan]:
        """Select spans from the database."""

        with self.session.begin() as session:
            query = session.query(self.Span)

            if where is not None:
                query = query.where(where)

            if page is not None:
                query = self._where_page(query=query, page=page)

            for orm_object in query.all():
                yield orm_object.write()

    def get_trace_record_ids(
        self,
        where: Optional[W] = None,
        page: Optional[core_db.PageSelect] = None,
    ) -> Iterable[str]:
        """Get the trace record ids matching the given query/page."""

        raise NotImplementedError

    def get_trace_record(self, record_id: str) -> Iterable[core_sem.TypedSpan]:
        """Select spans from the database that belong to the given record."""

        return self.get_spans(where=self.Span.record_ids.contains(record_id))
