import csv
from datetime import datetime
import logging
import os
import tempfile
import traceback
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import StatusCode
from trulens.connectors.snowflake import SnowflakeConnector
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
        def check_if_trulens_span(span: ReadableSpan) -> bool:
            if not span.attributes:
                return False

            return span.attributes.get("kind") == "SPAN_KIND_TRULENS"

        trulens_spans = list(filter(check_if_trulens_span, spans))

        if not trulens_spans:
            return SpanExportResult.SUCCESS

        if isinstance(self.connector, SnowflakeConnector):
            tmp_file_path = ""

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode="w", newline=""
                ) as tmp_file:
                    tmp_file_path = tmp_file.name
                    logger.debug(
                        f"Writing spans to the csv file: {tmp_file_path}"
                    )
                    writer = csv.writer(tmp_file)
                    writer.writerow(["span"])
                    for span in trulens_spans:
                        writer.writerow([span.to_json()])
                    logger.debug(
                        f"Spans written to the csv file: {tmp_file_path}"
                    )
            except Exception as e:
                logger.error(f"Error writing spans to the csv file: {e}")
                return SpanExportResult.FAILURE

            try:
                logger.debug("Uploading file to Snowflake stage")
                snowpark_session = self.connector.snowpark_session

                logger.debug("Creating Snowflake stage if it does not exist")
                snowpark_session.sql(
                    "CREATE STAGE IF NOT EXISTS trulens_spans"
                ).collect()

                logger.debug("Uploading the csv file to the stage")
                snowpark_session.sql(
                    f"PUT file://{tmp_file_path} @trulens_spans"
                ).collect()

            except Exception as e:
                print(f"Error uploading the csv file to the stage: {e}")
                traceback.print_exc()
                logger.error(f"Error uploading the csv file to the stage: {e}")
                return SpanExportResult.FAILURE

            try:
                logger.debug("Removing the temporary csv file")
                os.remove(tmp_file_path)
            except Exception as e:
                # Not returning failure here since the export was technically a success
                logger.error(f"Error removing the temporary csv file: {e}")

            return SpanExportResult.SUCCESS

        # For non-snowflake:
        try:
            events = list(map(self._construct_event, trulens_spans))
            self.connector.add_events(events)

        except Exception as e:
            logger.error(
                f"Error exporting spans to the database: {e}",
            )
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS
