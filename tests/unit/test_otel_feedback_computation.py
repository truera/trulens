"""
Tests for OTEL Feedback Computation.
"""

from typing import List

import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.feedback.computer import MinimalSpanInfo
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import Selector
from trulens.feedback.computer import _compute_feedback
from trulens.feedback.computer import _group_kwargs_by_selectors
from trulens.feedback.computer import compute_feedback_by_span_group
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.mock_otel_feedback_computation import (
    all_retrieval_span_attributes,
)
from tests.util.mock_otel_feedback_computation import feedback_function
from tests.util.otel_test_case import OtelTestCase

try:
    # These imports require optional dependencies to be installed.
    from trulens.apps.langchain import TruChain

    import tests.unit.test_otel_tru_chain
except Exception:
    pass


def _convert_events_to_MinimalSpanInfos(
    events: pd.DataFrame,
) -> List[MinimalSpanInfo]:
    ret = []
    for _, row in events.iterrows():
        span = MinimalSpanInfo()
        span.span_id = row["trace"]["span_id"]
        span.parent_span_id = row["record"]["parent_span_id"]
        if not span.parent_span_id:
            span.parent_span_id = None
        span.attributes = row["record_attributes"]
        ret.append(span)
    return ret


class _TestApp:
    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "span_group"})
    def call0(self, span_group: str, a0: int, b0: int, c0: int) -> List[int]:
        return [a0, b0, c0]

    @instrument()
    def call1(self, a1: int, b1: int, c1: int) -> List[int]:
        return [a1, b1, c1]

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.SPAN_GROUPS: [kwargs["span_group"]]
        }
    )
    def call2(self, span_group: str, a2: int, b2: int, c2: int) -> List[int]:
        self.call0(span_group, 2 * a2, 2 * b2, 2 * c2)
        return [a2, b2, c2]

    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "span_group"})
    def call4(
        self, span_group: str, a4: int, b4: int, c4: int, num_call3_calls: int
    ) -> List[int]:
        for _ in range(num_call3_calls):
            self.call0(span_group, 4 * a4, 4 * b4, 4 * c4)
        return [a4, b4, c4]

    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "span_group"})
    def call5(
        self, span_group: str, a5: int, b5: int, c5: int, num_call3_calls: int
    ) -> List[int]:
        for _ in range(num_call3_calls):
            self.call0(span_group, 5 * a5, 5 * b5, 5 * c5)
        return [a5, b5, c5]

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def query(self, question: str) -> str:
        # 1. Many attributes from one span that there are many of.
        self.call1(1, 1, 1)
        self.call1(2, 2, 2)
        self.call1(3, 3, 3)
        # 2. Attributes across spans where span groups are used.
        self.call2("10", 10, 10, 10)
        self.call2("11", 11, 11, 11)
        self.call2("12", 12, 12, 12)
        # 3. Attributes across spans where only one has more than one.
        self.call4("20", 20, 20, 20, 3)
        # 4. Attributes across spans where two have more than one (error case).
        self.call5("30", 30, 30, 30, 1)
        self.call5("30", 30, 30, 30, 1)
        # Return value is irrelevant.
        return "Kojikun"


@pytest.mark.optional
class TestOtelFeedbackComputation(OtelTestCase):
    def test_feedback_computation(self) -> None:
        # Create app.
        rag_chain = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            ground_truth_output="Like attention but with more heads.",
            main_method_args=("What is multi-headed attention?",),
        )
        TruSession().force_flush()
        # Compute feedback on record we just ingested.
        events = self._get_events()
        spans = _convert_events_to_MinimalSpanInfos(events)
        record_root = RecordGraphNode.build_graph(spans)
        _compute_feedback(
            record_root,
            "baby_grader",
            feedback_function,
            all_retrieval_span_attributes,
        )
        TruSession().force_flush()
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_feedback_computation__test_feedback_computation.csv"
        )

    def test_compute_feedback_by_span_group(self) -> None:
        # Create app.
        app = _TestApp()
        tru_app = TruApp(app, app_name="Test App", app_version="v1")
        # Record and invoke.
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_kwargs={
                "question": "What's the population of the capital of Japan?"
            },
        )
        TruSession().force_flush()
        # Compute feedback on record we just ingested.
        events = self._get_events()
        # 1. Many attributes from one span that there are many of.
        compute_feedback_by_span_group(
            events,
            "blah1",
            lambda a1, b1: 0.9 if a1 == b1 else 0.1,
            {
                "a1": Selector(
                    span_name="tests.unit.test_otel_feedback_computation._TestApp.call1",
                    span_attribute=f"{SpanAttributes.CALL.KWARGS}.a1",
                ),
                "b1": Selector(
                    span_name="tests.unit.test_otel_feedback_computation._TestApp.call1",
                    span_attribute=f"{SpanAttributes.CALL.KWARGS}.b1",
                ),
            },
        )
        # 2. Attributes across spans where span groups are used.
        compute_feedback_by_span_group(
            events,
            "blah2",
            lambda a2, a0: 0.9 if 2 * a2 == a0 else 0.1,
            {
                "a2": Selector(
                    span_name="tests.unit.test_otel_feedback_computation._TestApp.call2",
                    span_attribute=f"{SpanAttributes.CALL.KWARGS}.a2",
                ),
                "a0": Selector(
                    span_name="tests.unit.test_otel_feedback_computation._TestApp.call0",
                    span_attribute=f"{SpanAttributes.CALL.KWARGS}.a0",
                ),
            },
        )
        # 3. Attributes across spans where only one has more than one.
        # 4. Attributes across spans where two have more than one (error case).
        TruSession().force_flush()
        # Compare results to expected.
        events = self._get_events()
        eval_root_record_attributes = [
            curr["record_attributes"]
            for _, curr in events.iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        for i in range(3):
            self.assertEqual(
                eval_root_record_attributes[i][SpanAttributes.EVAL.METRIC_NAME],
                "blah1",
            )
            self.assertEqual(
                eval_root_record_attributes[i][SpanAttributes.EVAL_ROOT.RESULT],
                0.9,
            )
        for i in range(3, 6):
            self.assertEqual(
                eval_root_record_attributes[i][SpanAttributes.EVAL.METRIC_NAME],
                "blah2",
            )
            self.assertEqual(
                eval_root_record_attributes[i][SpanAttributes.EVAL_ROOT.RESULT],
                0.9,
            )

    def test__group_kwargs_by_selectors(self) -> None:
        # Test empty case.
        self.assertEqual([], _group_kwargs_by_selectors({}))
        # Test single selector case.
        self.assertEqual(
            [("input",)],
            _group_kwargs_by_selectors({
                "input": Selector(
                    span_type="a", span_name="b", span_attribute="c"
                )
            }),
        )
        # Test complex grouping.
        kwarg_to_selector = {
            "input1": Selector(
                span_type="a", span_name="b", span_attribute="x"
            ),
            "input2": Selector(
                span_type="a", span_name="b", span_attribute="x"
            ),
            "input3": Selector(
                span_type="c", span_name="d", span_attribute="x"
            ),
            "input4": Selector(span_type="c", span_attribute="x"),
            "input5": Selector(
                span_type="c", span_name="e", span_attribute="x"
            ),
        }
        self.assertEqual(
            [
                ("input1", "input2"),
                ("input3",),
                ("input4",),
                ("input5",),
            ],
            sorted(_group_kwargs_by_selectors(kwarg_to_selector)),
        )
