"""
Tests for OTEL instrument decorator.
"""

from unittest import TestCase
from unittest.mock import Mock

from trulens.core.otel.instrument import _resolve_attributes
from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
from trulens.experimental.otel_tracing.core.span import validate_attributes
from trulens.experimental.otel_tracing.core.span import validate_selector_name
from trulens.otel.semconv.trace import SpanAttributes


class TestOtelSpan(TestCase):
    def test__resolve_attributes(self):
        with self.subTest("None attributes"):
            self.assertEqual(
                {},
                _resolve_attributes(
                    None,
                    ret=None,
                    exception=None,
                    args=(),
                    all_kwargs={},
                ),
            )
        with self.subTest("Callable attributes"):
            attributes_callable = Mock(return_value={"key1": "value1"})
            self.assertEqual(
                {"key1": "value1"},
                _resolve_attributes(
                    attributes_callable,
                    ret=None,
                    exception=None,
                    args=(),
                    all_kwargs={},
                ),
            )
        with self.subTest("Dictionary attributes"):
            attributes_dict = {"key2": "value2", "key3": "return"}
            self.assertEqual(
                {"key2": "Kojikun", "key3": "Nolan"},
                _resolve_attributes(
                    attributes_dict,
                    ret="Nolan",
                    exception=None,
                    args=(),
                    all_kwargs={"value2": "Kojikun"},
                ),
            )

    def test_validate_selector_name(self):
        with self.subTest("No selector name"):
            self.assertEqual(
                validate_selector_name({}),
                {},
            )

        with self.subTest(
            f"Both {SpanAttributes.SELECTOR_NAME_KEY} and {SpanAttributes.SELECTOR_NAME} cannot be set."
        ):
            self.assertRaises(
                ValueError,
                validate_selector_name,
                {
                    SpanAttributes.SELECTOR_NAME_KEY: "key",
                    SpanAttributes.SELECTOR_NAME: "name",
                },
            )

        with self.subTest("Non-string"):
            self.assertRaises(
                ValueError,
                validate_selector_name,
                {
                    SpanAttributes.SELECTOR_NAME_KEY: 42,
                },
            )

        with self.subTest(f"Valid {SpanAttributes.SELECTOR_NAME}"):
            self.assertEqual(
                validate_selector_name({SpanAttributes.SELECTOR_NAME: "name"}),
                {SpanAttributes.SELECTOR_NAME_KEY: "name"},
            )

        with self.subTest(f"Valid {SpanAttributes.SELECTOR_NAME_KEY}"):
            self.assertEqual(
                validate_selector_name({
                    SpanAttributes.SELECTOR_NAME_KEY: "name"
                }),
                {SpanAttributes.SELECTOR_NAME_KEY: "name"},
            )

    def test_validate_attributes(self):
        with self.subTest("Empty attributes"):
            self.assertEqual(
                validate_attributes({}),
                {},
            )

        with self.subTest("Valid attributes"):
            self.assertEqual(
                validate_attributes({
                    "key1": "value1",
                    "key2": 123,
                    "key3": True,
                }),
                {
                    "key1": "value1",
                    "key2": 123,
                    "key3": True,
                },
            )

        with self.subTest("Invalid key type"):
            self.assertRaises(
                ValueError,
                validate_attributes,
                {
                    123: "value",
                },
            )

        with self.subTest("Span type should not be set in attributes"):
            self.assertRaises(
                ValueError,
                validate_attributes,
                {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.UNKNOWN,
                },
            )

    def test_set_user_defined_attributes(self):
        span = Mock()
        span_type = SpanAttributes.SpanType.UNKNOWN
        with self.subTest("Dictionary attributes"):
            attributes_dict = {"key2": "value2"}
            set_user_defined_attributes(
                span,
                span_type=span_type,
                attributes=attributes_dict,
            )
        with self.subTest("Invalid attributes"):
            attributes_invalid = {123: "value"}
            with self.assertRaises(ValueError):
                set_user_defined_attributes(
                    span,
                    span_type=span_type,
                    attributes=attributes_invalid,  # type: ignore
                )
