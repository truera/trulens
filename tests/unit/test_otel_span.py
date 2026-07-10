"""
Tests for OTEL instrument decorator.
"""

from unittest import TestCase
from unittest.mock import Mock

from trulens.core.otel.instrument import _resolve_attributes
from trulens.experimental.otel_tracing.core.span import (
    _convert_to_valid_span_attribute_type,
)
from trulens.experimental.otel_tracing.core.span import (
    set_genai_generation_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_genai_retrieval_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_genai_tool_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
from trulens.experimental.otel_tracing.core.span import validate_attributes
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes


class TestOtelSpan(TestCase):
    def test__resolve_attributes(self) -> None:
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

    def test_validate_attributes(self) -> None:
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

    def test_set_user_defined_attributes(self) -> None:
        span = Mock()
        with self.subTest("Dictionary attributes"):
            attributes_dict = {"key2": "value2"}
            set_user_defined_attributes(
                span,
                attributes=attributes_dict,
            )
        with self.subTest("Invalid attributes"):
            attributes_invalid = {123: "value"}
            with self.assertRaises(ValueError):
                set_user_defined_attributes(
                    span,
                    attributes=attributes_invalid,  # type: ignore
                )

    def test__convert_to_valid_span_attribute_type(self) -> None:
        class NonJsonifiable:
            def __str__(self):
                return "non-jsonifiable"

        with self.subTest("Boolean input returns original"):
            self.assertEqual(_convert_to_valid_span_attribute_type(True), True)
        with self.subTest("List of floats returns original"):
            float_list = [1.0, 2.0, 3.0]
            self.assertEqual(
                _convert_to_valid_span_attribute_type(float_list), float_list
            )
        with self.subTest("List of dicts returns jsonified version"):
            dict_list = [{"a": 1}, {"b": 2}]
            expected = '[{"a": 1}, {"b": 2}]'
            self.assertEqual(
                _convert_to_valid_span_attribute_type(dict_list), expected
            )
        with self.subTest(
            "List with non-jsonifiable object returns stringified list"
        ):
            obj = NonJsonifiable()
            self.assertEqual(
                _convert_to_valid_span_attribute_type([obj]),
                ["non-jsonifiable"],
            )
        with self.subTest("Non-jsonifiable object returns stringified object"):
            obj = NonJsonifiable()
            self.assertEqual(
                _convert_to_valid_span_attribute_type(obj), "non-jsonifiable"
            )


class TestGenAIHelpers(TestCase):
    """Unit tests for GenAI semantic convention dual-emit helpers."""

    def _make_span(self) -> Mock:
        span = Mock()
        span.set_attribute = Mock()
        return span

    # ------------------------------------------------------------------ #
    # set_genai_generation_attributes                                      #
    # ------------------------------------------------------------------ #

    def test_set_genai_generation_attributes_sets_expected_attributes(
        self,
    ) -> None:
        span = self._make_span()
        set_genai_generation_attributes(
            span,
            model="gpt-4o",
            input_tokens=10,
            output_tokens=20,
            temperature=0.7,
            provider_name="openai",
            operation_name="chat",
        )
        set_attribute_calls = {
            c.args[0]: c.args[1] for c in span.set_attribute.call_args_list
        }
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.REQUEST.MODEL], "gpt-4o"
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.USAGE.INPUT_TOKENS], 10
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.USAGE.OUTPUT_TOKENS], 20
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.REQUEST.TEMPERATURE], 0.7
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.SYSTEM.NAME], "openai"
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.OPERATION.NAME], "chat"
        )

    def test_set_genai_generation_attributes_skips_none_values(self) -> None:
        span = self._make_span()
        set_genai_generation_attributes(span)
        # Nothing should be set when all params are None.
        span.set_attribute.assert_not_called()

    def test_set_genai_generation_attributes_operation_name_none_not_emitted(
        self,
    ) -> None:
        span = self._make_span()
        set_genai_generation_attributes(
            span, model="claude-3", operation_name=None
        )
        set_attribute_calls = {
            c.args[0] for c in span.set_attribute.call_args_list
        }
        self.assertNotIn(GenAIAttributes.OPERATION.NAME, set_attribute_calls)
        self.assertIn(GenAIAttributes.REQUEST.MODEL, set_attribute_calls)

    # ------------------------------------------------------------------ #
    # set_genai_retrieval_attributes                                       #
    # ------------------------------------------------------------------ #

    def test_set_genai_retrieval_attributes_sets_expected_attributes(
        self,
    ) -> None:
        span = self._make_span()
        set_genai_retrieval_attributes(
            span,
            query_text="what is RAG?",
            documents=["doc1", "doc2"],
        )
        set_attribute_calls = {
            c.args[0]: c.args[1] for c in span.set_attribute.call_args_list
        }
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.RETRIEVAL.QUERY_TEXT],
            "what is RAG?",
        )
        self.assertIn(GenAIAttributes.RETRIEVAL.DOCUMENTS, set_attribute_calls)

    def test_set_genai_retrieval_attributes_skips_none_values(self) -> None:
        span = self._make_span()
        set_genai_retrieval_attributes(span)
        span.set_attribute.assert_not_called()

    def test_genai_retrieval_query_text_attribute_key(self) -> None:
        """Verify the attribute key matches the OTEL GenAI spec v1.36.0."""
        self.assertEqual(
            GenAIAttributes.RETRIEVAL.QUERY_TEXT, "gen_ai.retrieval.query.text"
        )

    # ------------------------------------------------------------------ #
    # set_genai_tool_attributes                                            #
    # ------------------------------------------------------------------ #

    def test_set_genai_tool_attributes_sets_expected_attributes(
        self,
    ) -> None:
        span = self._make_span()
        set_genai_tool_attributes(
            span,
            tool_name="search",
            call_arguments='{"query": "hello"}',
            call_result="result text",
        )
        set_attribute_calls = {
            c.args[0]: c.args[1] for c in span.set_attribute.call_args_list
        }
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.TOOL.NAME], "search"
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.TOOL.CALL_ARGUMENTS],
            '{"query": "hello"}',
        )
        self.assertEqual(
            set_attribute_calls[GenAIAttributes.TOOL.CALL_RESULT], "result text"
        )

    def test_set_genai_tool_attributes_skips_none_values(self) -> None:
        span = self._make_span()
        set_genai_tool_attributes(span)
        span.set_attribute.assert_not_called()
