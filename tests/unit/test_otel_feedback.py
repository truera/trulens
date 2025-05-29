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
