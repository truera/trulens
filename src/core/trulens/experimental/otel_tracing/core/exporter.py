from datetime import datetime
import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import StatusCode
from trulens.core.database import connector as core_connector
from trulens.core.schema import event as event_schema

logger = logging.getLogger(__name__)


def to_timestamp(timestamp: Optional[int]) -> datetime:
    if timestamp:
        return datetime.fromtimestamp(timestamp * 1e-9)

    return datetime.now()


class TruLensDBSpanExporter(SpanExporter):
    """
    Implementation of `SpanExporter` that flushes the spans to the database in the TruLens session.
    """

    connector: core_connector.DBConnector

    def __init__(self, connector: core_connector.DBConnector):
        self.connector = connector

    def _construct_event(self, span: ReadableSpan) -> event_schema.Event:
        context = span.get_span_context()
        parent = span.parent

        if context is None:
            raise ValueError("Span context is None")

        return event_schema.Event(
            event_id=str(context.span_id),
            record={
                "name": span.name,
                "kind": "SPAN_KIND_TRULENS",
                "parent_span_id": str(parent.span_id if parent else ""),
                "status": "STATUS_CODE_ERROR"
                if span.status.status_code == StatusCode.ERROR
                else "STATUS_CODE_UNSET",
            },
            record_attributes=span.attributes,
            record_type=event_schema.EventRecordType.SPAN,
            resource_attributes=span.resource.attributes,
            start_timestamp=to_timestamp(span.start_time),
            timestamp=to_timestamp(span.end_time),
            trace={
                "span_id": str(context.span_id),
                "trace_id": str(context.trace_id),
                "parent_id": str(parent.span_id if parent else ""),
            },
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            events = list(map(self._construct_event, spans))
            self.connector.add_events(events)

        except Exception as e:
            logger.error("Error exporting spans to the database: %s", e)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS
