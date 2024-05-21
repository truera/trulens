import json
from typing import Sequence, Union

from opentelemetry.sdk import trace as otsdk_trace
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor
from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import LargeBinary
from sqlalchemy import Text
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


class DBSpan(base):
    """Temporary db implementation for storing spans."""

    __tablename__ = "span"

    span_id = Column(Int64, primary_key=True)
    trace_id = Column(Int128, primary_key=True) # need double big int

    parent_span_id = Column(Int64)
    parent_trace_id = Column(Int128) # need double big int

    name = Column(VARCHAR(256))
    kind = Column(VARCHAR(256))

    #start_time = Column(BigInteger)
    #end_time = Column(BigInteger)

    status_code = Column(VARCHAR(32))
    status_description = Column(Text)

    attributes = Column(JSON) # JSON
    # events = Column(Text) # JSON
    # links = Column(Text) # JSON

    # attributes = relationship("DBSpanAttribute", backref=backref("span"))
    # events = relationship("DBSpanEvent", backref=backref("span"))
    # links = relationship("DBSpanLink", backref=backref("span"))

class TempDB():
    """Temporary database for storing spans."""

    def __init__(self, url: str):
        self.engine = create_engine(url)
        self.session = sessionmaker(self.engine)

        base.metadata.create_all(self.engine)

    def add_span(self, span: Union[OTSpan, otsdk_trace.ReadableSpan]):
        """Add a span to the database."""

        if isinstance(span, otsdk_trace.ReadableSpan):
            span: OTSpan = OTSpan.thaw(span)

        db_span = DBSpan(
            span_id = span.context.span_id.to_bytes(64),
            trace_id = span.context.trace_id.to_bytes(128),
            parent_span_id = span.parent_context.span_id.to_bytes(64) if span.parent_context else None,
            parent_trace_id = span.parent_context.trace_id.to_bytes(128) if span.parent_context else None,
            name = span.name,
            kind = span.kind.name,
            #start_time = span.start_time,
            #end_time = span.end_time,
            status_code = span.status.name,
            status_description = span.status_description,
            attributes = span.attributes,
            #events = json.dumps(span.events),
            #links = json.dumps(span.links)
        )

        with self.session.begin() as s:
            s.add(db_span)

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
