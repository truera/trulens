from typing import Optional

from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import GenAIEvents
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelSelector(OtelTestCase):
    def test_conversation_selectors(self) -> None:
        self.assertEqual(
            Selector.select_conversation(),
            Selector(
                conversation_level=True,
                conversation_attribute="records",
            ),
        )
        self.assertEqual(
            Selector.select_conversation_input().conversation_attribute,
            "input",
        )
        self.assertEqual(
            Selector.select_conversation_output().conversation_attribute,
            "output",
        )
        with self.assertRaises(ValueError):
            Selector(trace_level=True, conversation_level=True)

    def test__split_function_name(self) -> None:
        self.assertEqual(
            Selector._split_function_name("a.B.c"), ["a", "B", "c"]
        )
        self.assertEqual(
            Selector._split_function_name("a.py::B::c"), ["a.py", "B", "c"]
        )

    def test__matches_function_name(self) -> None:
        def _test_matches_function_name(
            selector_function_name: Optional[str],
            actual_function_name: Optional[str],
            match_expected: bool,
        ):
            selector = Selector(
                function_name=selector_function_name,
                span_name="span_name",
                span_attribute="Z",
            )
            if match_expected:
                self.assertTrue(
                    selector._matches_function_name(actual_function_name)
                )
            else:
                self.assertFalse(
                    selector._matches_function_name(actual_function_name)
                )

        _test_matches_function_name("AA.BB.CC", "AA.BB.CC", True)
        _test_matches_function_name("BB.CC", "AA.BB.CC", True)
        _test_matches_function_name("CC", "AA.BB.CC", True)
        _test_matches_function_name(None, "AA.BB.CC", True)
        _test_matches_function_name("AAA.BB.CC", "AA.BB.CC", False)
        _test_matches_function_name("AA.BB.CCC", "AA.BB.CC", False)
        _test_matches_function_name("A.BB.CC", "AA.BB.CC", False)
        _test_matches_function_name("B.CC", "AA.BB.CC", False)
        _test_matches_function_name("C", "AA.BB.CC", False)
        _test_matches_function_name("AA.BB.CC", None, False)
        selector = Selector(span_name="AA.BB.CC", span_attribute="Z")
        self.assertTrue(selector._matches_function_name("AA.BB.CC"))
        self.assertTrue(selector._matches_function_name("X"))
        self.assertTrue(selector._matches_function_name(None))

    def test_matches_spans(self) -> None:
        selector = Selector(
            function_name="AA.BB.CC",
            span_name="XX.YY.ZZ",
            span_type="span_type",
            span_attribute="Z",
        )
        self.assertTrue(
            selector.matches_span(
                "XX.YY.ZZ",
                {
                    SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                    SpanAttributes.SPAN_TYPE: "span_type",
                },
            )
        )
        self.assertFalse(
            selector.matches_span(
                "XX.YY.ZZ",
                {SpanAttributes.SPAN_TYPE: "span_type"},
            )
        )
        self.assertFalse(
            selector.matches_span(
                "XX.YY.ZZ",
                {SpanAttributes.CALL.FUNCTION: "AA.BB.CC"},
            )
        )
        self.assertFalse(
            selector.matches_span(
                None,
                {
                    SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                    SpanAttributes.SPAN_TYPE: "span_type",
                },
            )
        )
        self.assertTrue(
            Selector(function_name="CC", span_attribute="Z").matches_span(
                "XX.YY.ZZ",
                {
                    SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                    SpanAttributes.SPAN_TYPE: "span_type",
                },
            )
        )
        self.assertTrue(
            Selector(span_name="Y", span_attribute="Z").matches_span("Y", {})
        )

    def test_process_span(self) -> None:
        self.assertEqual(
            Selector(
                span_attributes_processor=lambda attributes: "z",
                function_name="X",
            ).process_span("1", {}),
            FeedbackFunctionInput(value="z", span_id="1"),
        )
        self.assertEqual(
            Selector(span_attribute="Z", function_name="X").process_span(
                "2", {}
            ),
            FeedbackFunctionInput(value=None, span_id="2", span_attribute="Z"),
        )
        self.assertEqual(
            Selector(span_attribute="Z", function_name="X").process_span(
                "3", {"Z": "z"}
            ),
            FeedbackFunctionInput(value="z", span_id="3", span_attribute="Z"),
        )
        self.assertEqual(
            Selector(
                function_attribute="return", function_name="X"
            ).process_span("4", {}),
            FeedbackFunctionInput(
                value=None,
                span_id="4",
                span_attribute=SpanAttributes.CALL.RETURN,
            ),
        )
        self.assertEqual(
            Selector(
                function_attribute="return", function_name="X"
            ).process_span("5", {SpanAttributes.CALL.RETURN: "z"}),
            FeedbackFunctionInput(
                value="z",
                span_id="5",
                span_attribute=SpanAttributes.CALL.RETURN,
            ),
        )
        self.assertEqual(
            Selector(function_attribute="arg1", function_name="X").process_span(
                "6", {}
            ),
            FeedbackFunctionInput(
                value=None,
                span_id="6",
                span_attribute=f"{SpanAttributes.CALL.KWARGS}.arg1",
            ),
        )
        self.assertEqual(
            Selector(function_attribute="arg1", function_name="X").process_span(
                "7", {f"{SpanAttributes.CALL.KWARGS}.arg1": "z"}
            ),
            FeedbackFunctionInput(
                value="z",
                span_id="7",
                span_attribute=f"{SpanAttributes.CALL.KWARGS}.arg1",
            ),
        )

    def test_span_event_attribute_is_an_extraction_mode(self) -> None:
        # Satisfies the "exactly one extraction mode" requirement on its own.
        selector = Selector(
            span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES
        )
        self.assertEqual(
            selector.span_event_attribute,
            GenAIEvents.EventAttributes.INPUT_MESSAGES,
        )

        with self.subTest("cannot combine with span_attribute"):
            with self.assertRaises(ValueError):
                Selector(
                    span_attribute="a",
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES,
                )

        with self.subTest("cannot combine with dataset_column"):
            with self.assertRaises(ValueError):
                Selector(
                    dataset_column="a",
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES,
                )

        with self.subTest("span_event_name requires span_event_attribute"):
            with self.assertRaises(ValueError):
                Selector(
                    span_attribute="a",
                    span_event_name=GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                )

    def test_process_span_reads_span_event_attributes(self) -> None:
        span_events = [
            {
                "name": GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                "timestamp": 1,
                "attributes": {
                    GenAIEvents.EventAttributes.INPUT_MESSAGES: "in",
                    GenAIEvents.EventAttributes.OUTPUT_MESSAGES: "out",
                },
            }
        ]

        with self.subTest("single match returns a scalar"):
            result = Selector(
                span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES
            ).process_span("1", {}, span_events)
            self.assertEqual(result.value, "in")
            # Provenance distinguishes an event attribute from a span attribute
            # of the same name.
            self.assertEqual(
                result.span_attribute,
                f"event/{GenAIEvents.EventAttributes.INPUT_MESSAGES}",
            )

        with self.subTest("span_event_name filters by event"):
            self.assertEqual(
                Selector(
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES,
                    span_event_name=GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                )
                .process_span("1", {}, span_events)
                .value,
                "in",
            )
            self.assertIsNone(
                Selector(
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES,
                    span_event_name="some.other.event",
                )
                .process_span("1", {}, span_events)
                .value,
            )

        with self.subTest("missing attribute yields None"):
            self.assertIsNone(
                Selector(span_event_attribute="nope")
                .process_span("1", {}, span_events)
                .value
            )

        with self.subTest("no events yields None"):
            self.assertIsNone(
                Selector(
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES
                )
                .process_span("1", {}, None)
                .value
            )

        with self.subTest("several matching events yield a list"):
            repeated = span_events + [
                {
                    "name": GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                    "timestamp": 2,
                    "attributes": {
                        GenAIEvents.EventAttributes.INPUT_MESSAGES: "in2"
                    },
                }
            ]
            self.assertEqual(
                Selector(
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES
                )
                .process_span("1", {}, repeated)
                .value,
                ["in", "in2"],
            )

        with self.subTest("span attributes are not consulted"):
            # An event selector must not silently fall back to a span attribute
            # that happens to share the key.
            self.assertIsNone(
                Selector(
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES
                )
                .process_span(
                    "1",
                    {GenAIEvents.EventAttributes.INPUT_MESSAGES: "from_span"},
                    None,
                )
                .value
            )
