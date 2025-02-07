import logging
import logging.handlers
import os
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

LOG_FILE = "/tmp/all_logs.txt"


def set_up_logging(
    log_level=logging.INFO, log_file=LOG_FILE, start_fresh: bool = True
):
    """
    Sets up the logging configuration for the main process and child processes.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).  Defaults to logging.INFO
        log_file: The file to log to.  Defaults to `/tmp/all_logs.txt`.
    """

    # Ensure the log file directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Delete the log file if it already exists.
    if start_fresh and os.path.exists(log_file):
        os.remove(log_file)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s"
    )

    # Create a handler that writes to the log file.  RotatingFileHandler is a good choice for production
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=5,
    )
    handler.setFormatter(formatter)

    # Get the root logger
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(handler)


class TruLensOTELSpanExporter(SpanExporter):
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
        set_up_logging(log_level=logging.DEBUG, start_fresh=False)
        logger = logging.getLogger(__name__)
        trulens_spans = list(filter(check_if_trulens_span, spans))

        try:
            events = list(map(construct_event, trulens_spans))
            logger.debug(
                f"Exporting {len(events)} events to the database:\n{events}"
            )
            self.connector.add_events(events)

        except Exception as e:
            logger.error(
                f"Error exporting spans to the database: {e}",
            )
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS
