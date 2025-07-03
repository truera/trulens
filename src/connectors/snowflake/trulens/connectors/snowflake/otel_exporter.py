from collections import defaultdict
import logging
import os
import tempfile
from typing import Sequence, Tuple

from google.protobuf.internal.encoder import _EncodeVarint
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.connectors.snowflake.dao.run import EvaluationPhase
from trulens.connectors.snowflake.dao.sql_utils import (
    clean_up_snowflake_identifier,
)
from trulens.core.database import connector as core_connector
from trulens.experimental.otel_tracing.core.exporter.utils import (
    check_if_trulens_span,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_readable_span_to_proto,
)
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class TruLensSnowflakeSpanExporter(SpanExporter):
    """
    Implementation of `SpanExporter` that flushes the spans in the TruLens session to a Snowflake Stage.
    """

    connector: SnowflakeConnector
    """
    The database connector used to export the spans.
    """

    def __init__(
        self,
        connector: core_connector.DBConnector,
        verify_via_dry_run: bool = True,
    ):
        if not isinstance(connector, SnowflakeConnector):
            raise ValueError("Provided connector is not a SnowflakeConnector")
        self.connector = connector  # type: ignore
        # Try to verify that this exporter will work as much as possible since
        # afterwards it'll run in its own thread and thus is hard to tell when
        # it fails.
        self.connector.snowpark_session.sql("SELECT 20240131").collect()
        if verify_via_dry_run:
            test_span = ReadableSpan(
                name="test_span",
                attributes={ResourceAttributes.APP_NAME: "test_app"},
            )

            res = self.export([test_span], dry_run=True)
            if res != SpanExportResult.SUCCESS:
                # This shouldn't happen since we should have been thrown errors.
                raise ValueError(
                    "OTEL Exporter failed dry run during initialization!"
                )

    def _export_to_snowflake(
        self, spans: Sequence[ReadableSpan], dry_run: bool
    ) -> SpanExportResult:
        """
        Exports a list of spans to a Snowflake stage as a protobuf file.
        This function performs the following steps:
        1. Writes the provided spans to a temporary protobuf file.
        2. Creates a Snowflake stage if it does not already exist.
        3. Uploads the temporary protobuf file to the Snowflake stage.
        4. Removes the temporary protobuf file.
        Args:
            spans: A sequence of spans to be exported.
            dry_run: Whether to do everything but run the ingestion SPROC.
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
                # TODO(otel, semconv, SNOW-2130988): Should have this in `span.resource.attributes`!
                span.attributes.get(ResourceAttributes.APP_NAME),
                span.attributes.get(ResourceAttributes.APP_VERSION),
                span.attributes.get(SpanAttributes.RUN_NAME),
                span.attributes.get(SpanAttributes.INPUT_RECORDS_COUNT),
            )
            app_and_run_info_to_spans[key].append(span)

        for (
            app_name,
            app_version,
            run_name,
            input_records_count,
        ), spans in app_and_run_info_to_spans.items():
            logger.debug(
                f"Logging {len(spans)} for app:{app_name} version:{app_version} run:{run_name}"
            )

            res = self._export_to_snowflake_stage_for_app_and_run(
                app_name,
                app_version,
                run_name,
                input_records_count,
                spans,
                dry_run,
            )
            if res == SpanExportResult.FAILURE:
                return res
        return SpanExportResult.SUCCESS

    @staticmethod
    def _write_spans_to_temp_file(
        spans: Sequence[ReadableSpan],
    ) -> Tuple[str, str]:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pb", mode="wb"
        ) as tmp_file:
            tmp_file_basename = os.path.basename(tmp_file.name)
            tmp_file_path = tmp_file.name
            logger.debug(f"Writing spans to the protobuf file: {tmp_file_path}")
            for span in spans:
                span_proto = convert_readable_span_to_proto(span)
                _EncodeVarint(tmp_file.write, span_proto.ByteSize())
                tmp_file.write(span_proto.SerializeToString())
            logger.debug(f"Spans written to the protobuf file: {tmp_file_path}")
        return tmp_file_path, tmp_file_basename

    @staticmethod
    def _upload_temp_file_to_stage(
        snowpark_session: Session, tmp_file_path: str
    ):
        stage_name = "trulens_spans"
        logger.debug("Creating Snowflake stage if it does not exist")
        snowpark_session.sql(
            f"CREATE TEMP STAGE IF NOT EXISTS {stage_name}"
        ).collect()
        logger.debug(f"Uploading file {tmp_file_path} to stage {stage_name}")
        snowpark_session.file.put(tmp_file_path, f"@{stage_name}")

    @staticmethod
    def _ingest_spans_from_stage(
        snowpark_session: Session,
        tmp_file_basename: str,
        app_name: str,
        app_version: str,
        run_name: str,
        input_records_count: int,
        dry_run: bool,
    ):
        database = clean_up_snowflake_identifier(
            snowpark_session.get_current_database()
        )
        schema = clean_up_snowflake_identifier(
            snowpark_session.get_current_schema()
        )

        sql_cmd = snowpark_session.sql(
            f"""
            CALL SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN(
                OBJECT_CONSTRUCT(
                    'object_name', ?,
                    'object_type', 'External Agent',
                    'object_version', ?
                ),
                OBJECT_CONSTRUCT(
                    'run_name', ?
                ),
                OBJECT_CONSTRUCT(
                    'type', 'stage_file',
                    'stage_file_path', ?,
                    'input_record_count', ?
                ),
                ARRAY_CONSTRUCT(),
                ARRAY_CONSTRUCT('{EvaluationPhase.INGESTION_MULTIPLE_BATCHES.value}')
            )
            """,
            params=[
                f"{database}.{schema}.{app_name.upper()}",
                app_version,
                run_name,
                f"@{database}.{schema}.trulens_spans/{tmp_file_basename}.gz",
                input_records_count,
            ],
        )
        if not dry_run:
            sql_cmd.collect_nowait()

    def _export_to_snowflake_stage_for_app_and_run(
        self,
        app_name: str,
        app_version: str,
        run_name: str,
        input_records_count: int,
        spans: Sequence[ReadableSpan],
        dry_run: bool,
    ) -> SpanExportResult:
        snowpark_session = self.connector.snowpark_session
        # Write spans to temp file.
        try:
            tmp_file_path, tmp_file_basename = self._write_spans_to_temp_file(
                spans
            )
        except Exception as e:
            logger.exception("Error writing spans to the protobuf file")
            if dry_run:
                raise e
            return SpanExportResult.FAILURE
        # Upload temp file to Snowflake stage.
        try:
            self._upload_temp_file_to_stage(snowpark_session, tmp_file_path)
        except Exception as e:
            logger.exception("Error uploading the protobuf file to the stage")
            if dry_run:
                raise e
            return SpanExportResult.FAILURE
        # Call the stored procedure to ingest the spans to the event table.
        try:
            self._ingest_spans_from_stage(
                snowpark_session,
                tmp_file_basename,
                app_name,
                app_version,
                run_name,
                input_records_count,
                dry_run,
            )
        except Exception as e:
            logger.error(f"Error running stored procedure to ingest spans: {e}")
            if dry_run:
                raise e
            return SpanExportResult.FAILURE
        # Remove the temp file.
        try:
            logger.debug("Removing the temporary protobuf file")
            os.remove(tmp_file_path)
        except Exception as e:
            # Not returning failure here since the export was technically a success
            logger.error(f"Error removing the temporary protobuf file: {e}")
            if dry_run:
                raise e
        # Successful return.
        return SpanExportResult.SUCCESS

    def export(
        self, spans: Sequence[ReadableSpan], dry_run: bool = False
    ) -> SpanExportResult:
        if dry_run:
            trulens_spans = spans
        else:
            trulens_spans = list(filter(check_if_trulens_span, spans))

        return self._export_to_snowflake(trulens_spans, dry_run)
