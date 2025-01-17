import unittest

from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.common.v1.common_pb2 import ArrayValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValueList
from trulens.experimental.otel_tracing.core.exporter.utils import (
    convert_to_any_value,
)


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


if __name__ == "__main__":
    unittest.main()
