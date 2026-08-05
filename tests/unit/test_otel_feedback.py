"""
Tests for OTEL Feedback methods.
"""

from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import Selector

from tests.util.otel_test_case import OtelTestCase


class TestOtelFeedback(OtelTestCase):
    def _mock_feedback_function_1(self, x: str) -> float:
        return 0.1

    def _mock_feedback_function_2(self, x: str, y: str) -> float:
        return 0.2

    def _mock_feedback_function_3(self, x: str, y: str, z: str) -> float:
        return 0.3

    def _mock_conversation_feedback(
        self, records: list, reference_topics: list
    ) -> float:
        return float(bool(records and reference_topics))

    def test_on_conversation_with_arguments(self) -> None:
        feedback = (
            Feedback(self._mock_conversation_feedback)
            .on_conversation()
            .with_arguments(reference_topics=["billing"])
        )
        self.assertEqual(
            feedback.selectors,
            {"records": Selector.select_conversation()},
        )
        self.assertEqual(
            feedback.implementation_kwargs,
            {"reference_topics": ["billing"]},
        )
        feedback.check_otel_selectors()
        self.assertEqual(feedback(records=[{"input": "hello"}]), 1.0)

    def test_conversation_selector_scope_validation(self) -> None:
        feedback = Feedback(
            self._mock_feedback_function_2,
            selectors={
                "x": Selector.select_conversation_input(),
                "y": Selector.select_record_output(),
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            feedback.check_otel_selectors()

    def test_on_input(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_input()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_input()}
        )
        feedback = Feedback(self._mock_feedback_function_2).on_input()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_input()}
        )
        feedback = Feedback(self._mock_feedback_function_3).on_input()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_input()}
        )

    def test_on_output(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_output()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_output()}
        )
        feedback = Feedback(self._mock_feedback_function_2).on_output()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_output()}
        )
        feedback = Feedback(self._mock_feedback_function_3).on_output()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_output()}
        )

    def test_on_input_output(self) -> None:
        with self.assertRaises(TypeError):
            Feedback(self._mock_feedback_function_1).on_input_output()
        feedback = Feedback(self._mock_feedback_function_2).on_input_output()
        self.assertEqual(
            feedback.selectors,
            {
                "x": Selector.select_record_input(),
                "y": Selector.select_record_output(),
            },
        )
        feedback = Feedback(self._mock_feedback_function_3).on_input_output()
        self.assertEqual(
            feedback.selectors,
            {
                "x": Selector.select_record_input(),
                "y": Selector.select_record_output(),
            },
        )

    def test_on_default(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_default()
        self.assertEqual(
            feedback.selectors, {"x": Selector.select_record_output()}
        )
        feedback = Feedback(self._mock_feedback_function_2).on_default()
        self.assertEqual(
            feedback.selectors,
            {
                "x": Selector.select_record_input(),
                "y": Selector.select_record_output(),
            },
        )
        with self.assertRaises(RuntimeError):
            Feedback(self._mock_feedback_function_3).on_default()
