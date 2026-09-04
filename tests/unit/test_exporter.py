import unittest

from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.common.v1.common_pb2 import ArrayValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValueList
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status
from opentelemetry.trace.status import StatusCode
from trulens.experimental.otel_tracing.core.exporter.utils import (
    construct_event,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_readable_span_to_proto,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_span_events,
)
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_to_any_value,
)
from trulens.otel.semconv.trace import GenAIEvents
from trulens.otel.semconv.trace import ResourceAttributes


class TestExporterUtils(unittest.TestCase):
    def test_convert_to_any_value(self):
        with self.subTest("String value"):
            value = "test_string"
            any_value = convert_to_any_value(value)
            self.assertEqual(any_value.string_value, value)

        with self.subTest("Boolean value"):
            value = True
            any_value = convert_to_any_value(value)
            self.assertEqual(any_value.bool_value, value)

        with self.subTest("Integer value"):
            value = 123
            any_value = convert_to_any_value(value)
            self.assertEqual(any_value.int_value, value)

        with self.subTest("Float value"):
            value = 123.45
            any_value = convert_to_any_value(value)
            self.assertAlmostEqual(any_value.double_value, value)

        with self.subTest("Bytes value"):
            value = b"test_bytes"
            any_value = convert_to_any_value(value)
            self.assertEqual(any_value.bytes_value, value)

        with self.subTest("List value"):
            value = ["test_string", 123, 123.45, True]
            any_value = convert_to_any_value(value)
            self.assertEqual(
                any_value.array_value,
                ArrayValue(
                    values=[
                        AnyValue(string_value="test_string"),
                        AnyValue(int_value=123),
                        AnyValue(double_value=123.45),
                        AnyValue(bool_value=True),
                    ]
                ),
            )

        with self.subTest("Dictionary value"):
            value = {
                "key1": "value1",
                "key2": 123,
                "key3": 123.45,
                "key4": True,
            }
            any_value = convert_to_any_value(value)
            self.assertEqual(
                any_value.kvlist_value,
                KeyValueList(
                    values=[
                        KeyValue(
                            key="key1", value=AnyValue(string_value="value1")
                        ),
                        KeyValue(key="key2", value=AnyValue(int_value=123)),
                        KeyValue(
                            key="key3", value=AnyValue(double_value=123.45)
                        ),
                        KeyValue(key="key4", value=AnyValue(bool_value=True)),
                    ]
                ),
            )

        with self.subTest("Unsupported type"):
            value = set([1, 2, 3])
            with self.assertRaises(
                ValueError, msg="Unsupported value type: <class 'set'>"
            ):
                convert_to_any_value(value)

        with self.subTest("Tuple value"):
            value = ("test_string", 123, 123.45, True)
            any_value = convert_to_any_value(value)
            self.assertEqual(
                any_value.array_value,
                ArrayValue(
                    values=[
                        AnyValue(string_value="test_string"),
                        AnyValue(int_value=123),
                        AnyValue(double_value=123.45),
                        AnyValue(bool_value=True),
                    ]
                ),
            )


def _readable_span(events=()) -> ReadableSpan:
    """A minimal ReadableSpan carrying the attributes construct_event needs."""
    return ReadableSpan(
        name="test_span",
        context=SpanContext(trace_id=0x1234, span_id=0x5678, is_remote=False),
        parent=None,
        resource=Resource.create({}),
        attributes={ResourceAttributes.APP_NAME: "test_app"},
        events=events,
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.UNSET),
        start_time=1_000_000_000,
        end_time=2_000_000_000,
    )


_INFERENCE_EVENT = Event(
    name=GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
    attributes={
        GenAIEvents.EventAttributes.INPUT_MESSAGES: '[{"role": "user"}]',
        GenAIEvents.EventAttributes.OUTPUT_MESSAGES: '[{"role": "assistant"}]',
    },
    timestamp=1_500_000_000,
)


class TestSpanEventSerialization(unittest.TestCase):
    """Span events carry GenAI message content and must survive export."""

    def test_convert_span_events_serializes_name_timestamp_and_attributes(self):
        events = convert_span_events(_readable_span(events=[_INFERENCE_EVENT]))

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0]["name"], GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS
        )
        self.assertEqual(events[0]["timestamp"], 1_500_000_000)
        self.assertEqual(
            events[0]["attributes"][GenAIEvents.EventAttributes.INPUT_MESSAGES],
            '[{"role": "user"}]',
        )
        self.assertEqual(
            events[0]["attributes"][
                GenAIEvents.EventAttributes.OUTPUT_MESSAGES
            ],
            '[{"role": "assistant"}]',
        )

    def test_convert_span_events_returns_empty_without_events(self):
        self.assertEqual(convert_span_events(_readable_span()), [])

    def test_construct_event_records_span_events(self):
        event = construct_event(_readable_span(events=[_INFERENCE_EVENT]))

        self.assertIn("events", event.record)
        self.assertEqual(len(event.record["events"]), 1)
        self.assertEqual(
            event.record["events"][0]["attributes"][
                GenAIEvents.EventAttributes.INPUT_MESSAGES
            ],
            '[{"role": "user"}]',
        )

    def test_construct_event_omits_events_key_when_span_has_none(self):
        # Content capture is opt-in, so most spans have no events. The key must
        # be absent rather than empty so persisted rows are unchanged.
        event = construct_event(_readable_span())

        self.assertNotIn("events", event.record)
        self.assertEqual(
            set(event.record),
            {"name", "kind", "parent_span_id", "status"},
        )

    def test_proto_conversion_populates_events(self):
        span_proto = convert_readable_span_to_proto(
            _readable_span(events=[_INFERENCE_EVENT])
        )

        self.assertEqual(len(span_proto.events), 1)
        proto_event = span_proto.events[0]
        self.assertEqual(
            proto_event.name, GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS
        )
        self.assertEqual(proto_event.time_unix_nano, 1_500_000_000)
        self.assertEqual(
            {kv.key for kv in proto_event.attributes},
            {
                GenAIEvents.EventAttributes.INPUT_MESSAGES,
                GenAIEvents.EventAttributes.OUTPUT_MESSAGES,
            },
        )

    def test_proto_conversion_has_no_events_when_span_has_none(self):
        span_proto = convert_readable_span_to_proto(_readable_span())

        self.assertEqual(len(span_proto.events), 0)
