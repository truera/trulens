import json
from typing import Sequence, Union

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace import SpanProcessor
from sqlaclhemy import Integer
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import VARCHAR
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

# Temporary database design seperate from the main database.
base = declarative_base()

class DBSpan(base):
    """Temporary db implementation for storing spans."""

    __tablename__ = "span"

    span_id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, primary_key=True)

    parent_span_id = Column(Integer)
    parent_trace_id = Column(Integer)

    name = Column(VARCHAR(255))
    kind = Column(VARCHAR(255))

    start_time = Column(Integer)
    end_time = Column(Integer)

    status_code = Column(VARCHAR(16))
    status_description = Column(Text)

    attributes = Column(Text) # JSON
    events = Column(Text) # JSON
    links = Column(Text) # JSON

    # attributes = relationship("DBSpanAttribute", backref=backref("span"))
    # events = relationship("DBSpanEvent", backref=backref("span"))
    # links = relationship("DBSpanLink", backref=backref("span"))

class TempDB():
    """Temporary database for storing spans."""

    def __init__(self, url: str):
        self.engine = create_engine(url)
        self.session = sessionmaker(self.engine)

        base.metadata.create_all(self.engine)

    def add_span(self, span: Union[Span, otsdk_trace.ReadableSpan]):
        """Add a span to the database."""

        if isinstance(span, Span):
            span: otsdk_trace.ReadableSpan = span.freeze()

        db_span = DBSpan(
            span_id = span.span_id,
            trace_id = span.trace_id,
            parent_span_id = span.parent_span_id,
            parent_trace_id = span.parent_trace_id,
            name = span.name,
            kind = span.kind,
            start_time = span.start_time,
            end_time = span.end_time,
            status_code = span.status,
            status_description = span.status_description,
            attributes = json.dumps(span.attributes),
            events = json.dumps(span.events),
            links = json.dumps(span.links)
        )
        self.session.add(db_span)

class AlchemyExporter(export.SpanExporter):
    """Exporter for spans to a SQLAlchemy database."""

    def __init__(self, db: TempDB):
        self.db = db

    def export(self, spans: Sequence[ReadableSpan]) -> None:
        """Export spans to the database."""

        for span in spans:
            self.db.add_span(span)

    def shutdown(self):
        """Shutdown the exporter."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter.
        
        Not needed for this exporter."""
        
        return True
