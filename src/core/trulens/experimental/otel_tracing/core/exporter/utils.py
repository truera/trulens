from datetime import datetime
import logging
from typing import Any, Optional

from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.common.v1.common_pb2 import ArrayValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValueList
from opentelemetry.proto.trace.v1.trace_pb2 import Span as SpanProto
from opentelemetry.proto.trace.v1.trace_pb2 import Status
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import ResourceAttributes

logger = logging.getLogger(__name__)


def convert_to_any_value(value: Any) -> AnyValue:
    """
    Converts a given value to an AnyValue object.
    This function takes a value of various types (str, bool, int, float, bytes,
    list, dict) and converts it into an AnyValue object. If the value is a list
    or a dictionary, it recursively converts the elements or key-value pairs.
    Tuples are converted into lists.
    Args:
        value:
            The value to be converted. It can be of type str, bool, int, float,
            bytes, list, tuple, or dict.
    Returns:
        The converted AnyValue object.
    """
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
    """
    Converts a ReadableSpan object to a protobuf object for a Span.
    Args:
        span: The span to be converted.
    Returns:
        SpanProto: The converted span in SpanProto format.
    """
    span_proto = SpanProto(
        trace_id=span.context.trace_id.to_bytes(16, "big")
        if span.context
        else b"",
        span_id=span.context.span_id.to_bytes(8, "big")
        if span.context
        else b"",
        parent_span_id=span.parent.span_id.to_bytes(8, "big")
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
        status=Status(code=Status.StatusCode.STATUS_CODE_UNSET),
    )
    # TODO(otel): Remove this once the Snowflake backend no longer uses this!
    span_proto.attributes.append(
        KeyValue(key="name", value=convert_to_any_value(span.name))
    )
    return span_proto


def to_timestamp(timestamp: Optional[int]) -> datetime:
    """
    Utility function for converting OTEL timestamps to datetime objects.
    """
    if timestamp:
        return datetime.fromtimestamp(timestamp * 1e-9)

    return datetime.now()


def check_if_trulens_span(span: ReadableSpan) -> bool:
    """
    Check if a given span is a TruLens span.
    This function checks the attributes of the provided span to determine if it
    contains a TruLens-specific attribute, identified by the presence of
    `SpanAttributes.RECORD_ID`.
    Args:
        span: The span to be checked.
    Returns:
        True if the span contains the TruLens-specific attribute, False otherwise.
    """
    if not span.attributes:
        return False
    # TODO(otel, semconv): Should have this in `span.resource.attributes`!
    app_name = span.attributes.get(ResourceAttributes.APP_NAME)
    return bool(app_name)


def construct_event(span: ReadableSpan) -> event_schema.Event:
    context = span.get_span_context()
    parent = span.parent

    if context is None:
        raise ValueError("Span context is None")

    ret = event_schema.Event(
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
    # TODO(otel, semconv, SNOW-2130988):
    # This is only a workaround for now until we can directly put these into the
    # resource attributes.
    for k in [
        ResourceAttributes.APP_ID,
        ResourceAttributes.APP_NAME,
        ResourceAttributes.APP_VERSION,
    ]:
        if k in ret.record_attributes:
            ret.resource_attributes[k] = ret.record_attributes[k]
    return ret
