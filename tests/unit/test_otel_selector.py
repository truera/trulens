from typing import Optional

from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelSelector(OtelTestCase):
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
            selector.matches_span({
                SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                SpanAttributes.SPAN_TYPE: "span_type",
                "name": "XX.YY.ZZ",
            })
        )
        self.assertFalse(
            selector.matches_span({
                SpanAttributes.SPAN_TYPE: "span_type",
                "name": "XX.YY.ZZ",
            })
        )
        self.assertFalse(
            selector.matches_span({
                SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                "name": "XX.YY.ZZ",
            })
        )
        self.assertFalse(
            selector.matches_span({
                SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                SpanAttributes.SPAN_TYPE: "span_type",
            })
        )
        self.assertTrue(
            Selector(function_name="CC", span_attribute="Z").matches_span({
                SpanAttributes.CALL.FUNCTION: "AA.BB.CC",
                SpanAttributes.SPAN_TYPE: "span_type",
                "name": "XX.YY.ZZ",
            })
        )
        self.assertTrue(
            Selector(span_name="Y", span_attribute="Z").matches_span({
                "name": "Y"
            })
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
