import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.core.database import connector as core_connector
from trulens.experimental.otel_tracing.core.exporter.utils import (
    check_if_trulens_span,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    construct_event,
)

logger = logging.getLogger(__name__)


class TruLensOtelSpanExporter(SpanExporter):
    """
    Implementation of `SpanExporter` that flushes the spans in the TruLens session to the connector.
    """

    connector: core_connector.DBConnector
    """
    The database connector used to export the spans.
    """

    def __init__(self, connector: core_connector.DBConnector):
        self.connector = connector

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        trulens_spans = list(filter(check_if_trulens_span, spans))

        try:
            events = list(map(construct_event, trulens_spans))
            self.connector.add_events(events)

        except Exception as e:
            logger.error(
                f"Error exporting spans to the database: {e}",
            )
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS
