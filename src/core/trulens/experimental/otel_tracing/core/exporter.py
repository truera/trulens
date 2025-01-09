from datetime import datetime
import logging
import os
import tempfile
from typing import Any, Optional, Sequence

from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.common.v1.common_pb2 import ArrayValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValueList
from opentelemetry.proto.trace.v1.trace_pb2 import Span as SpanProto
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import StatusCode
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.database import connector as core_connector
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


def convert_to_any_value(value: Any) -> AnyValue:
    if isinstance(value, tuple):
        value = list(value)
    any_value = AnyValue()

    if isinstance(value, str):
        any_value.string_value = value
    elif isinstance(value, bool):
        any_value.bool_value = value
    elif isinstance(value, int):
        any_value.int_value = value
    elif isinstance(value, float):
        any_value.double_value = value
    elif isinstance(value, bytes):
        any_value.bytes_value = value
    elif isinstance(value, list):
        array_value = ArrayValue()
        for item in value:
            array_value.values.append(convert_to_any_value(item))
        any_value.array_value.CopyFrom(array_value)
    elif isinstance(value, dict):
        kv_list = KeyValueList()
        for k, v in value.items():
            kv = KeyValue(key=k, value=convert_to_any_value(v))
            kv_list.values.append(kv)
        any_value.kvlist_value.CopyFrom(kv_list)
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

    return any_value


def convert_readable_span_to_proto(span: ReadableSpan) -> SpanProto:
    span_proto = SpanProto(
        trace_id=span.context.trace_id.to_bytes(16, byteorder="big")
        if span.context
        else b"",
        span_id=span.context.span_id.to_bytes(8, byteorder="big")
        if span.context
        else b"",
        parent_span_id=span.parent.span_id.to_bytes(8, byteorder="big")
        if span.parent
        else b"",
        name=span.name,
        kind=SpanProto.SpanKind.SPAN_KIND_INTERNAL,
        start_time_unix_nano=span.start_time if span.start_time else 0,
        end_time_unix_nano=span.end_time if span.end_time else 0,
        attributes=[
            KeyValue(key=k, value=convert_to_any_value(v))
            for k, v in span.attributes.items()
        ]
        if span.attributes
        else None,
    )
    return span_proto


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
                "kind": SpanProto.SpanKind.SPAN_KIND_INTERNAL,
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

            return bool(
                span.attributes.get(SpanAttributes.GENERATED_BY_INSTRUMENTATION)
            )

        trulens_spans = list(filter(check_if_trulens_span, spans))

        if not trulens_spans:
            return SpanExportResult.SUCCESS

        if isinstance(self.connector, SnowflakeConnector):
            tmp_file_path = ""

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pb", mode="wb"
                ) as tmp_file:
                    tmp_file_path = tmp_file.name
                    logger.debug(
                        f"Writing spans to the protobuf file: {tmp_file_path}"
                    )

                    for span in trulens_spans:
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
                snowpark_session = self.connector.snowpark_session

                logger.debug("Creating Snowflake stage if it does not exist")
                snowpark_session.sql(
                    "CREATE STAGE IF NOT EXISTS trulens_spans"
                ).collect()

                logger.debug("Uploading the protobuf file to the stage")
                snowpark_session.sql(
                    f"PUT file://{tmp_file_path} @trulens_spans"
                ).collect()

            except Exception as e:
                logger.error(
                    f"Error uploading the protobuf file to the stage: {e}"
                )
                return SpanExportResult.FAILURE

            try:
                logger.debug("Removing the temporary protobuf file")
                os.remove(tmp_file_path)
            except Exception as e:
                # Not returning failure here since the export was technically a success
                logger.error(f"Error removing the temporary protobuf file: {e}")

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
