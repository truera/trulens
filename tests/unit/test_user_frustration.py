"""Unit tests for user_frustration conversation metric."""

from typing import Dict, Optional, Tuple
from unittest import TestCase

from trulens.core import Metric
from trulens.feedback import llm_provider
from trulens.feedback.templates import conversation as templates_conversation


class MockLLMProvider(llm_provider.LLMProvider):
    """Mock LLM provider that returns configurable scores."""

    model_config = {"extra": "allow"}

    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None
    last_temperature: Optional[float] = None
    mock_score_response: str = "Score: 2\nReason: Test reason"

    def __init__(self, **kwargs):
        super().__init__(
            endpoint=None,
            model_engine="mock-model",
            **kwargs,
        )

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ) -> str:
        """Return configurable mock response."""
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    self.last_system_prompt = msg.get("content", "")
                elif msg.get("role") == "user":
                    self.last_user_prompt = msg.get("content", "")
        elif prompt:
            self.last_system_prompt = prompt
        return self.mock_score_response


class TestUserFrustration(TestCase):
    """Test user_frustration metric with real generate_score mapping."""

    def setUp(self):
        self.provider = MockLLMProvider()
        self.sample_conversation = [
            {"input": "What is my balance?", "output": "You have $100."},
            {"input": "That's wrong, check again.", "output": "You have $0."},
            {"input": "What is my balance?", "output": "You have $0."},
        ]

    def test_template_registered(self):
        """UserFrustration template exists and is properly configured."""
        self.assertTrue(hasattr(templates_conversation, "UserFrustration"))
        self.assertEqual(
            templates_conversation.UserFrustration.output_space, "LIKERT_0_3"
        )
        self.assertIn("UserFrustration", templates_conversation.__all__)

    def test_score_mapping_severe_frustration(self):
        """Template raw score 0 (severe frustration) → normalized 1.0 (high frustration)."""
        self.provider.mock_score_response = "Score: 0\nReason: User gave up"
        score = self.provider.user_frustration(self.sample_conversation)
        # Template: 0=severe frustration → reversed to 1.0 (high frustration score)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_score_mapping_no_frustration(self):
        """Template raw score 3 (no frustration) → normalized 0.0 (low frustration)."""
        self.provider.mock_score_response = "Score: 3\nReason: User satisfied"
        score = self.provider.user_frustration(self.sample_conversation)
        # Template: 3=no frustration → reversed to 0.0 (low frustration score)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_score_mapping_mild_frustration(self):
        """Template raw score 2 (mild frustration) → normalized ~0.33."""
        self.provider.mock_score_response = "Score: 2\nReason: Minor issue"
        score = self.provider.user_frustration(self.sample_conversation)
        # Template: 2=mild frustration → reversed to ~0.33
        self.assertAlmostEqual(score, 1.0 / 3.0, places=2)

    def test_temperature_forwarding(self):
        """Temperature parameter is passed to generate_score."""
        self.provider.user_frustration(self.sample_conversation, temperature=0.7)
        self.assertEqual(self.provider.last_temperature, 0.7)

    def test_user_turns_in_prompt(self):
        """User turns are included in the prompt sent to LLM."""
        self.provider.user_frustration(self.sample_conversation)
        self.assertIsNotNone(self.provider.last_user_prompt)
        self.assertIn("What is my balance?", self.provider.last_user_prompt)
        self.assertIn("That's wrong", self.provider.last_user_prompt)

    def test_additional_instructions_forwarded(self):
        """additional_instructions are appended to system prompt."""
        custom = "Pay special attention to sarcasm."
        self.provider.user_frustration(
            self.sample_conversation, additional_instructions=custom
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(custom, self.provider.last_system_prompt)

    def test_with_cot_reasons(self):
        """user_frustration_with_cot_reasons returns score and reasons."""
        self.provider.mock_score_response = "Score: 1\nReason: User frustrated"
        score, reasons = self.provider.user_frustration_with_cot_reasons(
            self.sample_conversation
        )
        # Template: 1=moderate frustration → reversed to ~0.67
        self.assertAlmostEqual(score, 2.0 / 3.0, places=2)
        self.assertIsInstance(reasons, dict)
        # COT template should be appended to user prompt
        self.assertIn("COT REASONS", self.provider.last_user_prompt)

    def test_metric_wrapper(self):
        """Metric wrapper correctly invokes user_frustration."""
        self.provider.mock_score_response = "Score: 0\nReason: Gave up"
        metric = Metric(
            implementation=self.provider.user_frustration,
            name="User Frustration",
        )
        # Metric should be callable on conversation
        result = metric.evaluate(self.sample_conversation)
        self.assertAlmostEqual(result, 1.0, places=2)


if __name__ == "__main__":
    import unittest
    unittest.main()
