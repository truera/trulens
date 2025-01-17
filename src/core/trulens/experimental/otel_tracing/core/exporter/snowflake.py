import logging
import os
import tempfile
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.database import connector as core_connector
from trulens.experimental.otel_tracing.core.exporter.utils import (
    check_if_trulens_span,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_readable_span_to_proto,
)

logger = logging.getLogger(__name__)


class TruLensSnowflakeSpanExporter(SpanExporter):
    """
    Implementation of `SpanExporter` that flushes the spans in the TruLens session to a Snowflake Stage.
    """

    connector: core_connector.DBConnector
    """
    The database connector used to export the spans.
    """

    def __init__(self, connector: core_connector.DBConnector):
        self.connector = connector

    def _export_to_snowflake_stage(
        self, spans: Sequence[ReadableSpan]
    ) -> SpanExportResult:
        """
        Exports a list of spans to a Snowflake stage as a protobuf file.
        This function performs the following steps:
        1. Writes the provided spans to a temporary protobuf file.
        2. Creates a Snowflake stage if it does not already exist.
        3. Uploads the temporary protobuf file to the Snowflake stage.
        4. Removes the temporary protobuf file.
        Args:
            spans (Sequence[ReadableSpan]): A sequence of spans to be exported.
        Returns:
            SpanExportResult: The result of the export operation, either SUCCESS or FAILURE.
        """
        if not isinstance(self.connector, SnowflakeConnector):
            return SpanExportResult.FAILURE

        # Avoid uploading empty files to the stage
        if not spans:
            return SpanExportResult.SUCCESS

        snowpark_session = self.connector.snowpark_session
        tmp_file_path = ""

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pb", mode="wb"
            ) as tmp_file:
                tmp_file_path = tmp_file.name
                logger.debug(
                    f"Writing spans to the protobuf file: {tmp_file_path}"
                )

                for span in spans:
                    span_proto = convert_readable_span_to_proto(span)
                    tmp_file.write(span_proto.SerializeToString())
                logger.debug(
                    f"Spans written to the protobuf file: {tmp_file_path}"
                )
        except Exception as e:
            logger.error(f"Error writing spans to the protobuf file: {e}")
            return SpanExportResult.FAILURE

        try:
            logger.debug("Uploading file to Snowflake stage")

            logger.debug("Creating Snowflake stage if it does not exist")
            snowpark_session.sql(
                "CREATE TEMP STAGE IF NOT EXISTS trulens_spans"
            ).collect()

            logger.debug("Uploading the protobuf file to the stage")
            snowpark_session.sql(
                f"PUT file://{tmp_file_path} @trulens_spans"
            ).collect()

        except Exception as e:
            logger.error(f"Error uploading the protobuf file to the stage: {e}")
            return SpanExportResult.FAILURE

        try:
            logger.debug("Removing the temporary protobuf file")
            os.remove(tmp_file_path)
        except Exception as e:
            # Not returning failure here since the export was technically a success
            logger.error(f"Error removing the temporary protobuf file: {e}")

        return SpanExportResult.SUCCESS

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        trulens_spans = list(filter(check_if_trulens_span, spans))

        return self._export_to_snowflake_stage(trulens_spans)
