from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import RECORD_ROOT_INPUT
from trulens.core.feedback.selector import RECORD_ROOT_OUTPUT

from tests.util.otel_test_case import OtelTestCase

"""
Tests for OTEL Feedback methods.
"""


class TestOtelFeedback(OtelTestCase):
    def _mock_feedback_function_1(self, x: str) -> float:
        return 0.1

    def _mock_feedback_function_2(self, x: str, y: str) -> float:
        return 0.2

    def _mock_feedback_function_3(self, x: str, y: str, z: str) -> float:
        return 0.3

    def test_on_input(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_input()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_INPUT})
        feedback = Feedback(self._mock_feedback_function_2).on_input()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_INPUT})
        feedback = Feedback(self._mock_feedback_function_3).on_input()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_INPUT})

    def test_on_output(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_output()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_OUTPUT})
        feedback = Feedback(self._mock_feedback_function_2).on_output()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_OUTPUT})
        feedback = Feedback(self._mock_feedback_function_3).on_output()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_OUTPUT})

    def test_on_input_output(self) -> None:
        with self.assertRaises(TypeError):
            Feedback(self._mock_feedback_function_1).on_input_output()
        feedback = Feedback(self._mock_feedback_function_2).on_input_output()
        self.assertEqual(
            feedback.selectors,
            {"x": RECORD_ROOT_INPUT, "y": RECORD_ROOT_OUTPUT},
        )
        feedback = Feedback(self._mock_feedback_function_3).on_input_output()
        self.assertEqual(
            feedback.selectors,
            {"x": RECORD_ROOT_INPUT, "y": RECORD_ROOT_OUTPUT},
        )

    def test_on_default(self) -> None:
        feedback = Feedback(self._mock_feedback_function_1).on_default()
        self.assertEqual(feedback.selectors, {"x": RECORD_ROOT_OUTPUT})
        feedback = Feedback(self._mock_feedback_function_2).on_default()
        self.assertEqual(
            feedback.selectors,
            {"x": RECORD_ROOT_INPUT, "y": RECORD_ROOT_OUTPUT},
        )
        with self.assertRaises(RuntimeError):
            Feedback(self._mock_feedback_function_3).on_default()
