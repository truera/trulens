from typing import Iterable, List, Optional

import sqlalchemy as sa
from trulens.core.database import base as core_db
from trulens.core.database import sqlalchemy as core_sqlalchemy
from trulens.experimental.otel_tracing.core import otel as core_otel
from trulens.experimental.otel_tracing.core.database import base as otel_core_db
from trulens.experimental.otel_tracing.core.database import orm as otel_core_orm


class _SQLAlchemyDB(core_sqlalchemy.SQLAlchemyDB):
    """EXPERIMENTAL(otel_tracing): Adds span API to the SQLAlchemy DB API."""

    Q = core_sqlalchemy.SQLAlchemyDB.Q

    def _where_span_index(self, query: Q, index: otel_core_db.SpanIndex) -> Q:
        """Adds where clauses to the given Span query based on the given index."""

        self.orm: (
            otel_core_orm.SpanORM
        )  # assume self.orm was patched with otel_tracing additions

        if index.index is None and (
            index.span_id is None or index.trace_id is None
        ):
            raise ValueError(
                "Span index must have either index or span_id and trace_id."
            )

        if index.index is not None:
            query = query.where(self.orm.Span.index == index.index)

        if index.span_id is not None:
            query = query.where(self.orm.Span.span_id == index.span_id)

        if index.trace_id is not None:
            query = query.where(self.orm.Span.trace_id == index.trace_id)

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
            query = query.where(self.orm.Span.index > page.after_index)

        if page.before_index is not None:
            query = query.where(self.orm.Span.index < page.before_index)

        if page.after_created_timestamp:
            query = query.where(
                self.orm.Span.created_timestamp > page.after_created_timestamp
            )

        if page.before_created_timestamp:
            query = query.where(
                self.orm.Span.created_timestamp < page.before_created_timestamp
            )

        if page.after_updated_timestamp:
            query = query.where(
                self.orm.Span.updated_timestamp > page.after_updated_timestamp
            )

        if page.before_updated_timestamp:
            query = query.where(
                self.orm.Span.updated_timestamp < page.before_updated_timestamp
            )

        return query

    def insert_span(self, span: core_otel.Span) -> otel_core_db.SpanIndex:
        """Insert a span into the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            orm = self.orm.Span.parse(span)
            session.merge(orm)

            return otel_core_db.SpanIndex(
                index=orm.index, span_id=orm.span_id, trace_id=orm.trace_id
            )

    def batch_insert_span(
        self, spans: Iterable[core_otel.Span]
    ) -> List[otel_core_db.SpanIndex]:
        """Insert a batch of spans into the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            orms = [self.orm.Span.parse(span) for span in spans]
            session.add_all(orms)

            return [
                otel_core_db.SpanIndex(
                    index=orm.index, span_id=orm.span_id, trace_id=orm.trace_id
                )
                for orm in orms
            ]

    def delete_span(self, index: otel_core_db.SpanIndex) -> None:
        """Delete a span from the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            query = self._where_span_index(
                query=session.query(self.orm.Span), index=index
            ).first()
            session.delete(query)

    def delete_spans(
        self,
        query: Optional[Q] = None,
        page: Optional[core_db.PageSelect] = None,
    ) -> None:
        """Delete spans from the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            if query is None:
                query = session.query(self.orm.Span)

            if page is not None:
                query = self._where_page(query=query, page=page)

            session.delete(query)

    def get_span(
        self, index: otel_core_db.SpanIndex
    ) -> Optional[core_otel.Span]:
        """Get a span from the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            query = self._where_span_index(
                query=session.query(self.orm.Span), index=index
            ).first()
            orm = query.one_or_none()

            if orm is None:
                return None

            return self.orm.Span.write()

    def get_spans(
        self,
        query: Optional[Q] = None,
        page: Optional[core_db.PageSelect] = None,
    ) -> Iterable[core_otel.Span]:
        """Select spans from the database."""

        self.orm: otel_core_orm.SpanORM

        with self.session.begin() as session:
            if query is None:
                query = session.query(self.orm.Span)

            if page is not None:
                query = self._where_page(query=query, page=page)

            for orm in query.all():
                yield orm.write()
