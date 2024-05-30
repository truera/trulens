"""
TODO: Figure out how we can reduce the number of representations of the same data:
- OTSpan and subclasses
- DBSpan for storing in sqlalchemy
- ReadableSpan for exporting
"""

import json
from typing import Sequence, Union

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import status as trace_status
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
import sqlalchemy
from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import LargeBinary
from sqlalchemy import Text
from sqlalchemy import UUID
from sqlalchemy import VARCHAR
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

from trulens_eval.trace import flatten_lensed_attributes
from trulens_eval.trace import OTSpan

# Temporary database design seperate from the main database.
base = declarative_base()

Int128 = LargeBinary(128)
Int64 = LargeBinary(64)
TTraceID = Int128
TSpanID = Int64

class DBSpan(base):
    """Temporary db implementation for storing spans."""

    __tablename__ = "span"

    span_id = Column(TSpanID, primary_key=True)
    trace_id = Column(TTraceID, primary_key=True) # need double big int

    parent_span_id = Column(TSpanID)
    parent_trace_id = Column(TTraceID) # need double big int

    name = Column(VARCHAR(256))
    kind = Column(VARCHAR(256))

    start_time = Column(BigInteger)
    end_time = Column(BigInteger)

    status_code = Column(VARCHAR(32))
    status_description = Column(Text)

    attributes = Column(JSON)
    events = Column(JSON)

    def freeze(self) -> ReadableSpan:
        """Freeze the span into a ReadableSpan."""
        return otsdk_trace.ReadableSpan(
            name=self.name,
            context=ot_span.SpanContext(
                trace_id=int.from_bytes(self.trace_id),
                span_id=int.from_bytes(self.span_id),
                is_remote=False
            ),
            parent=ot_span.SpanContext(
                trace_id=int.from_bytes(self.parent_trace_id),
                span_id=int.from_bytes(self.parent_span_id),
                is_remote=False
            ) if self.parent_span_id else None,
            resource = None,
            attributes=self.attributes,
            events=self.events,
            links=[], # todo
            kind=ot_trace.SpanKind[self.kind],
            instrumentation_info = None,
            status=ot_span.Status(trace_status.StatusCode[self.status_code], self.status_description),
            start_time=self.start_time,
            end_time=self.end_time,
            instrumentation_scope = None
        )

    def thaw(self) -> OTSpan:
        return OTSpan.thaw(self.freeze())

    # attributes = relationship("DBSpanAttribute", backref=backref("span"))
    # events = relationship("DBSpanEvent", backref=backref("span"))


class DBLink(base):
    """Temporary db implementation for storing links."""

    __tablename__ = "link"

    # TODO: ForeignKey("span.trace_id"), ForeignKey("span.span_id")
    source_span_id = Column(TSpanID, primary_key=True)
    source_trace_id = Column(TTraceID, primary_key=True)
    target_span_id = Column(TSpanID, primary_key=True)
    target_trace_id = Column(TTraceID, primary_key=True)

    attributes = Column(JSON)

    # TODO:
    #source = relationship(
    #    "DBSpan",
    #    backref=backref("links"),
    #    foreign_keys=[source_span_id, source_trace_id],
    #    primaryjoin="DBLink.source_span_id == DBSpan.span_id & DBLink.source_trace_id == DBSpan.trace_id"
    #)

    
class TraceDB():
    """Temporary database for storing spans."""

    def __init__(self, url: str):
        self.engine = create_engine(url)
        self.session = sessionmaker(self.engine)

        base.metadata.create_all(self.engine)

    def __enter__(self):
        self.session_context: sqlalchemy.orm.session.Session = self.session.begin().__enter__()
        return self.session_context

    def __exit__(self, *args, **kwargs):
        self._ensure_session()

        self.session_context.commit()

        self.session_context.__exit__(*args, **kwargs)
        self.session_context = None

    def _ensure_session(self):
        if self.session_context is None:
            raise RuntimeError("Session not started")

    def add_span(self, span: Union[OTSpan, otsdk_trace.ReadableSpan]):
        """Add a span to the database."""

        self._ensure_session()

        if isinstance(span, otsdk_trace.ReadableSpan):
            span: OTSpan = OTSpan.thaw(span)

        db_span = DBSpan(
            span_id = span.context.span_id.to_bytes(64),
            trace_id = span.context.trace_id.to_bytes(128),
            parent_span_id = span.parent_context.span_id.to_bytes(64) if span.parent_context else None,
            parent_trace_id = span.parent_context.trace_id.to_bytes(128) if span.parent_context else None,
            name = span.name,
            kind = span.kind.name,
            start_time = span.start_timestamp,
            end_time = span.end_timestamp,
            status_code = span.status.name,
            status_description = span.status_description,
            attributes = span.attributes,
            events = span.events
        )

        print("adding span", span.context.span_id % 100)
        self.session_context.add(db_span)

        for target_context, attrs in span.links.items():
            print("Adding link", span.context.span_id % 100, "->", target_context.span_id % 100)
            self.session_context.add(DBLink(
                source_trace_id = span.context.trace_id.to_bytes(128),
                source_span_id = span.context.span_id.to_bytes(64),
                target_trace_id = target_context.trace_id.to_bytes(128),
                target_span_id = target_context.span_id.to_bytes(64),
                attributes = attrs
            ))

class AlchemyExporter(export.SpanExporter):
    """Exporter for spans to a SQLAlchemy database."""

    def __init__(self, db: TraceDB):
        self.db = db
        self.to_export = []

    def export(self, spans: Sequence[Union[ReadableSpan, OTSpan]]) -> None:
        """Export spans to the database."""

        for span in spans:
            self.to_export.append(span)

    def shutdown(self):
        """Shutdown the exporter."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter.
        
        Writes all spans to the database.
        """
                
        with self.db:
            for span in self.to_export:
                self.db.add_span(span)

        self.to_export = []

        return True
