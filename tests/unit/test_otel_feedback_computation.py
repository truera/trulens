"""
Tests for OTEL Feedback Computation.
"""

import gc
import time
from typing import Callable, List
import weakref

import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core.feedback import Feedback
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.feedback.computer import MinimalSpanInfo
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback
from trulens.feedback.computer import _flatten_inputs
from trulens.feedback.computer import _group_kwargs_by_selectors
from trulens.feedback.computer import _remove_already_computed_feedbacks
from trulens.feedback.computer import _validate_unflattened_inputs
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
    def call3(
        self, span_group: str, a3: int, b3: int, c3: int, num_call3_calls: int
    ) -> List[int]:
        for _ in range(num_call3_calls):
            self.call0(span_group, 3 * a3, 3 * b3, 3 * c3)
        return [a3, b3, c3]

    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "span_group"})
    def call4(
        self, span_group: str, a4: int, b4: int, c4: int, num_call3_calls: int
    ) -> List[int]:
        for _ in range(num_call3_calls):
            self.call0(span_group, 4 * a4, 4 * b4, 4 * c4)
        return [a4, b4, c4]

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
        self.call3("20", 20, 20, 20, 3)
        # 4. Attributes across spans where two have more than one (error case).
        self.call4("30", 30, 30, 30, 1)
        self.call4("30", 30, 30, 30, 1)
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
        events = self._get_events()
        # Compute feedback on record we just ingested.
        get_selector = lambda s: Selector(
            span_name=f"tests.unit.test_otel_feedback_computation._TestApp.call{s[1]}",
            span_attribute=f"{SpanAttributes.CALL.KWARGS}.{s}",
        )
        # Case 1. Two attributes from one function that has multiple (three)
        #         invocations.
        compute_feedback_by_span_group(
            events,
            "blah1",
            lambda a1, b1: 0.9 if a1 == b1 else 0.1,
            {"a1": get_selector("a1"), "b1": get_selector("b1")},
        )
        # Case 2. Attributes across functions with span groups.
        compute_feedback_by_span_group(
            events,
            "blah2",
            lambda a2, a0: 0.9 if 2 * a2 == a0 else 0.1,
            {"a2": get_selector("a2"), "a0": get_selector("a0")},
        )
        # Case 3. Attributes across functions with span groups where one
        #         function is invoked once and the other multiple (three)
        #         times.
        compute_feedback_by_span_group(
            events,
            "blah3",
            lambda a3, a0: 0.9 if 3 * a3 == a0 else 0.1,
            {"a3": get_selector("a3"), "a0": get_selector("a0")},
        )
        # Case 4. Attributes across functions where both functions are invoked
        #         more than once (error case).
        with self.assertRaisesRegex(
            ValueError,
            "^No feedbacks were computed!$",
        ):
            compute_feedback_by_span_group(
                events,
                "blah4",
                lambda a4, a0: 0.9 if 4 * a4 == a0 else 0.1,
                {"a4": get_selector("a4"), "a0": get_selector("a0")},
            )
        # Compare results to expected.
        TruSession().force_flush()
        events = self._get_events()
        eval_root_record_attributes = [
            curr["record_attributes"]
            for _, curr in events.iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        expected_case_number = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.assertEqual(
            len(expected_case_number),
            len(eval_root_record_attributes),
        )
        for curr in eval_root_record_attributes:
            self.assertEqual(curr[SpanAttributes.EVAL_ROOT.SCORE], 0.9)
        for i in range(len(expected_case_number)):
            self.assertEqual(
                eval_root_record_attributes[i][SpanAttributes.EVAL.METRIC_NAME],
                f"blah{expected_case_number[i]}",
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

    def test__validate_unflattened_inputs(self) -> None:
        kwarg_groups = [("a",), ("b",)]
        record_id = "record_id1"
        span_group = "span_group1"
        record_ids_with_record_roots = [record_id]
        feedback_name = "feedback1"
        # Happy path.
        unflattened_inputs = {
            (record_id, span_group): {
                ("a",): [{"a": 1}, {"a": 1.1}],
                ("b",): [{"b": 2}],
            }
        }
        res = _validate_unflattened_inputs(
            unflattened_inputs,
            kwarg_groups,
            record_ids_with_record_roots,
            feedback_name,
        )
        self.assertEqual(
            {
                (record_id, span_group): {
                    ("a",): [{"a": 1}, {"a": 1.1}],
                    ("b",): [{"b": 2}],
                }
            },
            res,
        )
        # Missing inputs are removed.
        unflattened_inputs = {(record_id, span_group): {("a",): [{"a": 1}]}}
        res = _validate_unflattened_inputs(
            unflattened_inputs,
            kwarg_groups,
            record_ids_with_record_roots,
            feedback_name,
        )
        self.assertEqual({}, res)
        # Ambiguous inputs are removed and have error message.
        unflattened_inputs = {
            (record_id, span_group): {
                ("a",): [{"a": 1}, {"a", 1.1}],
                ("b",): [{"b": 2}, {"b": 2.2}],
            }
        }
        res = _validate_unflattened_inputs(
            unflattened_inputs,
            kwarg_groups,
            record_ids_with_record_roots,
            feedback_name,
        )
        self.assertEqual({}, res)
        # Records without a record root are removed and have error message.
        invalid_record_id = "invalid_record_id"
        unflattened_inputs = {
            (invalid_record_id, span_group): {
                ("a",): [{"a": 1}, {"a": 1.1}],
                ("b",): [{"b": 2}],
            }
        }
        res = _validate_unflattened_inputs(
            unflattened_inputs,
            kwarg_groups,
            record_ids_with_record_roots,
            feedback_name,
        )
        self.assertEqual({}, res)

    def test__flatten_inputs(self) -> None:
        record_id = "record_id1"
        span_group = "span_group1"
        unflattened_inputs = {
            (record_id, span_group): {
                ("a",): [{"a": 1}, {"a": 1.1}],
                ("b",): [{"b": 2}],
            }
        }
        res = _flatten_inputs(unflattened_inputs)
        self.assertEqual(
            [
                (record_id, span_group, {"a": 1, "b": 2}),
                (record_id, span_group, {"a": 1.1, "b": 2}),
            ],
            res,
        )

    def test__remove_already_computed_feedbacks(self) -> None:
        events = pd.DataFrame({
            "record_attributes": [
                {
                    SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.EVAL_ROOT,
                    SpanAttributes.EVAL.METRIC_NAME: "feedback1",
                    SpanAttributes.RECORD_ID: "record_id1",
                    SpanAttributes.EVAL_ROOT.SPAN_GROUP: "span_group1",
                    SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID + ".a": "span_id1a",
                    SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID + ".b": "span_id1b",
                    SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE
                    + ".a": "span_attribute1a",
                    SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE
                    + ".b": "span_attribute1b",
                },
            ]
        })
        flattened_inputs = [
            (
                "record_id1",
                "span_group1",
                {
                    "a": FeedbackFunctionInput(
                        span_id="span_id1a", span_attribute="span_attribute1a"
                    ),
                    "b": FeedbackFunctionInput(
                        span_id="span_id1b", span_attribute="span_attribute1b"
                    ),
                },
            ),
            (
                "record_id1",
                "span_group1",
                {
                    "a": FeedbackFunctionInput(
                        span_id="span_id1a", span_attribute="span_attribute1a"
                    ),
                    "b": FeedbackFunctionInput(
                        span_id="span_id2b", span_attribute="span_attribute1b"
                    ),
                },
            ),
        ]
        res = _remove_already_computed_feedbacks(
            events, "feedback1", flattened_inputs
        )
        self.assertEqual([flattened_inputs[1]], res)

    def _create_invoked_app_with_custom_feedback(self) -> TruApp:
        # Create feedback function.
        def custom(input: str, output: str) -> float:
            if (
                input == "What is multi-headed attention?"
                and output == "This is a mocked response for prompt 0."
            ):
                return 0.42
            return 0.0

        f_custom = Feedback(custom, name="custom").on({
            "input": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
            "output": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
            ),
        })

        # Create app.
        rag_chain = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
            feedbacks=[f_custom],
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What is multi-headed attention?",),
        )
        TruSession().force_flush()
        return tru_recorder

    def test_custom_feedback(self) -> None:
        tru_recorder = self._create_invoked_app_with_custom_feedback()
        # Compute feedback on record we just ingested.
        num_events = len(self._get_events())
        tru_recorder.compute_feedbacks()
        TruSession().force_flush()
        # Compare results to expected.
        events = self._get_events()
        self.assertEqual(num_events + 1, len(events))
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
        )
        record_root_span_id = events.iloc[0]["trace"]["span_id"]
        last_event_record_attributes = events.iloc[-1]["record_attributes"]
        self.assertEqual(
            SpanAttributes.SpanType.EVAL_ROOT,
            last_event_record_attributes[SpanAttributes.SPAN_TYPE],
        )
        self.assertEqual(
            "custom",
            last_event_record_attributes[SpanAttributes.EVAL.METRIC_NAME],
        )
        self.assertEqual(
            0.42,
            last_event_record_attributes[SpanAttributes.EVAL_ROOT.SCORE],
        )
        self.assertEqual(
            record_root_span_id,
            last_event_record_attributes[
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID + ".input"
            ],
        )
        self.assertEqual(
            SpanAttributes.RECORD_ROOT.INPUT,
            last_event_record_attributes[
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE + ".input"
            ],
        )
        self.assertEqual(
            record_root_span_id,
            last_event_record_attributes[
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID + ".output"
            ],
        )
        self.assertEqual(
            SpanAttributes.RECORD_ROOT.OUTPUT,
            last_event_record_attributes[
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE + ".output"
            ],
        )
        # Check that when trying to compute feedbacks again, nothing happens.
        old_num_events = len(self._get_events())
        tru_recorder.compute_feedbacks(
            raise_error_on_no_feedbacks_computed=False
        )
        TruSession().force_flush()
        new_num_events = len(self._get_events())
        self.assertEqual(old_num_events, new_num_events)

    def test_evaluator(self) -> None:
        tru_recorder = self._create_invoked_app_with_custom_feedback()
        num_events = len(self._get_events())
        tru_recorder.start_evaluator()
        # Wait for there to be a feedback computed.
        self._wait(lambda: len(self._get_events()) > num_events)
        events = self._get_events()
        self.assertEqual(num_events + 1, len(events))
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
        )
        # Ensure thread to be stopped when app is garbage collected.
        tru_recorder_ref = weakref.ref(tru_recorder)
        evaluator_ref = weakref.ref(tru_recorder._evaluator)
        thread_ref = weakref.ref(tru_recorder._evaluator._thread)
        del tru_recorder
        gc.collect()
        self.assertCollected(tru_recorder_ref)
        self._wait(lambda: thread_ref() is None)
        self.assertCollected(evaluator_ref)
        self.assertCollected(thread_ref)

    @staticmethod
    def _wait(
        f: Callable[[], bool],
        timeout_in_seconds: float = 30,
        cooldown_in_seconds: float = 1,
    ):
        start_time = time.time()
        while time.time() - start_time < timeout_in_seconds:
            if f():
                return
            time.sleep(cooldown_in_seconds)
        raise TimeoutError(
            f"Timed out waiting for condition to be met after {timeout_in_seconds} seconds."
        )
