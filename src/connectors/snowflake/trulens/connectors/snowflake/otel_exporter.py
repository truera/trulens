from collections import defaultdict
import logging
import os
import tempfile
from typing import Sequence

from google.protobuf.internal.encoder import _EncodeVarint
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
from trulens.otel.semconv.trace import SpanAttributes

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

        app_and_run_info_to_spans = defaultdict(list)
        for span in spans:
            key = (
                span.attributes.get(SpanAttributes.APP_NAME),
                span.attributes.get(SpanAttributes.APP_VERSION),
                span.attributes.get(SpanAttributes.RUN_NAME),
            )
            app_and_run_info_to_spans[key].append(span)

        for (
            app_name,
            app_version,
            run_name,
        ), spans in app_and_run_info_to_spans.items():
            res = self._export_to_snowflake_stage_for_app_and_run(
                app_name, app_version, run_name, spans
            )
            if res == SpanExportResult.FAILURE:
                return res
        return SpanExportResult.SUCCESS

    def _export_to_snowflake_stage_for_app_and_run(
        self,
        app_name: str,
        app_version: str,
        run_name: str,
        spans: Sequence[ReadableSpan],
    ) -> SpanExportResult:
        snowpark_session = self.connector.snowpark_session
        # Write spans to temp file.
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pb", mode="wb"
            ) as tmp_file:
                tmp_file_basename = os.path.basename(tmp_file.name)
                tmp_file_path = tmp_file.name
                logger.debug(
                    f"Writing spans to the protobuf file: {tmp_file_path}"
                )
                for span in spans:
                    span_proto = convert_readable_span_to_proto(span)
                    _EncodeVarint(tmp_file.write, span_proto.ByteSize())
                    tmp_file.write(span_proto.SerializeToString())
                logger.debug(
                    f"Spans written to the protobuf file: {tmp_file_path}"
                )
        except Exception as e:
            logger.error(f"Error writing spans to the protobuf file: {e}")
            return SpanExportResult.FAILURE
        # Upload temp file to Snowflake stage.
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
        # Call the stored procedure to ingest the spans to the event table.
        try:
            snowpark_session.sql(
                "ALTER SESSION SET TRACE_LEVEL=ALWAYS"
            ).collect()
            snowpark_session.sql(f"""
                -- TODO(this_pr): the name of the SPROC is going to change hopefully...
                CALL YUZHAO.AI_OBS.DELIMITED_INGEST_AI_OBS_SPANS(
                    BUILD_SCOPED_FILE_URL(
                        @{snowpark_session.get_current_database()}.{snowpark_session.get_current_schema()}.trulens_spans,
                        '{tmp_file_basename}.gz'
                    ),
                    '{app_name}',
                    '{app_version}',
                    '{run_name}'
                )
            """).collect()
        except Exception as e:
            logger.error(f"Error running stored procedure to ingest spans: {e}")
            return SpanExportResult.FAILURE
        # Remove the temp file.
        try:
            logger.debug("Removing the temporary protobuf file")
            os.remove(tmp_file_path)
        except Exception as e:
            # Not returning failure here since the export was technically a success
            logger.error(f"Error removing the temporary protobuf file: {e}")
        # Successful return.
        return SpanExportResult.SUCCESS

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        trulens_spans = list(filter(check_if_trulens_span, spans))

        return self._export_to_snowflake_stage(trulens_spans)
