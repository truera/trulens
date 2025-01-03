"""
Tests for OTEL instrument decorator.
"""

from unittest import TestCase
from unittest import main
from unittest.mock import Mock

from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
from trulens.experimental.otel_tracing.core.span import validate_attributes
from trulens.experimental.otel_tracing.core.span import validate_selector_name
from trulens.otel.semconv.trace import SpanAttributes


class TestOTELSpan(TestCase):
    def testValidateSelectorName(self):
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

    def testValidateAttributes(self):
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

    def testSetUserDefinedAttributes(self):
        span = Mock()
        span_type = SpanAttributes.SpanType.UNKNOWN

        with self.subTest("Callable attributes"):

            def attributes_callable(ret, func_exception, *args, **kwargs):
                return {"key1": "value1"}

            set_user_defined_attributes(
                span,
                span_type=span_type,
                args=(),
                kwargs={},
                ret=None,
                func_exception=None,
                attributes=attributes_callable,
            )
            span.set_attribute.assert_any_call("trulens.unknown.key1", "value1")

        with self.subTest("Dictionary attributes"):
            attributes_dict = {"key2": "value2"}

            set_user_defined_attributes(
                span,
                span_type=span_type,
                args=(),
                kwargs={},
                ret=None,
                func_exception=None,
                attributes=attributes_dict,
            )
            span.set_attribute.assert_any_call("trulens.unknown.key2", "value2")

        with self.subTest("Invalid attributes"):
            attributes_invalid = {123: "value"}

            with self.assertRaises(ValueError):
                set_user_defined_attributes(
                    span,
                    span_type=span_type,
                    args=(),
                    kwargs={},
                    ret=None,
                    func_exception=None,
                    attributes=attributes_invalid,  # type: ignore
                )


if __name__ == "__main__":
    main()
