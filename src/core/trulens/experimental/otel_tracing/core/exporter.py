from datetime import datetime
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.core.database import connector as core_connector
from trulens.core.schema import event as event_schema


class TruLensDBSpanExporter(SpanExporter):
    """
    Implementation of :class:`SpanExporter` that flushes the spans to the database in the TruLens session.
    """

    connector: core_connector.DBConnector

    def __init__(self, connector: core_connector.DBConnector):
        self.connector = connector

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            context = span.get_span_context()

            if context is None:
                continue

            event = event_schema.Event(
                event_id=str(context.span_id),
                record=span.attributes,
                record_attributes=span.attributes,
                record_type="SPAN",
                resource_attributes=span.resource.attributes,
                start_timestamp=datetime.fromtimestamp(
                    span.start_time / pow(10, 9)
                )
                if span.start_time
                else datetime.now(),
                timestamp=datetime.fromtimestamp(span.end_time / pow(10, 9))
                if span.end_time
                else datetime.now(),
                trace={
                    "trace_id": str(context.trace_id),
                    "parent_id": str(context.span_id),
                },
            )
            self.connector.add_event(event)
            print(f"adding event {str(context.span_id)}")
            print(event)
        return SpanExportResult.SUCCESS

    """Immediately export all spans"""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
