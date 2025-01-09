from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.common.v1.common_pb2 import ArrayValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.common.v1.common_pb2 import KeyValueList
import pytest
from trulens.experimental.otel_tracing.core.exporter import convert_to_any_value


def test_convert_to_any_value_string():
    value = "test_string"
    any_value = convert_to_any_value(value)
    assert any_value.string_value == value


def test_convert_to_any_value_bool():
    value = True
    any_value = convert_to_any_value(value)
    assert any_value.bool_value == value


def test_convert_to_any_value_int():
    value = 123
    any_value = convert_to_any_value(value)
    assert any_value.int_value == value


def test_convert_to_any_value_float():
    value = 123.45
    any_value = convert_to_any_value(value)
    assert any_value.double_value == pytest.approx(value)


def test_convert_to_any_value_bytes():
    value = b"test_bytes"
    any_value = convert_to_any_value(value)
    assert any_value.bytes_value == value


def test_convert_to_any_value_list():
    value = ["test_string", 123, 123.45, True]
    any_value = convert_to_any_value(value)
    assert any_value.array_value == ArrayValue(
        values=[
            AnyValue(string_value="test_string"),
            AnyValue(int_value=123),
            AnyValue(double_value=123.45),
            AnyValue(bool_value=True),
        ]
    )


def test_convert_to_any_value_dict():
    value = {"key1": "value1", "key2": 123, "key3": 123.45, "key4": True}
    any_value = convert_to_any_value(value)
    assert any_value.kvlist_value == KeyValueList(
        values=[
            KeyValue(key="key1", value=AnyValue(string_value="value1")),
            KeyValue(key="key2", value=AnyValue(int_value=123)),
            KeyValue(key="key3", value=AnyValue(double_value=123.45)),
            KeyValue(key="key4", value=AnyValue(bool_value=True)),
        ]
    )


def test_convert_to_any_value_unsupported_type():
    value = set([1, 2, 3])
    with pytest.raises(
        ValueError, match="Unsupported value type: <class 'set'>"
    ):
        convert_to_any_value(value)
