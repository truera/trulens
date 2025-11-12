"""Unit tests for feedback function criteria and custom_instructions parameters."""

import os

os.environ["TRULENS_OTEL_TRACING"] = "0"

from typing import Dict, Optional, Tuple
import unittest
from unittest import TestCase

from trulens.core import Feedback
from trulens.feedback import llm_provider
from trulens.feedback.v2 import feedback as feedback_v2


class MockLLMProvider(llm_provider.LLMProvider):
    """Mock LLM provider for testing criteria and custom_instructions."""

    model_config = {"extra": "allow"}  # Allow additional test attributes

    # Declare test tracking fields as optional Pydantic fields
    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None

    def __init__(self, **kwargs):
        # Initialize with minimal required fields
        super().__init__(
            endpoint=None,  # No actual endpoint needed for testing
            model_engine="mock-model",
            **kwargs,
        )

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ) -> str:
        """Mock chat completion that records prompts."""
        # Capture the system prompt for inspection
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    self.last_system_prompt = msg.get("content", "")
                elif msg.get("role") == "user":
                    self.last_user_prompt = msg.get("content", "")
        elif prompt:
            self.last_system_prompt = prompt

        # Return a mock response with a score
        return "Score: 2\nReason: This is a test reason."

    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """Mock generate_score that captures prompts."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 0.67  # Normalized 2/3

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Mock generate_score_and_reasons that captures prompts."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 0.67, {"reason": "Test reason"}


class TestFeedbackCriteria(TestCase):
    """Test that criteria parameter works correctly."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_helpfulness_default_criteria(self):
        """Test that helpfulness uses default criteria when none provided."""
        result = self.provider.helpfulness_with_cot_reasons(
            text="This is helpful text"
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], dict)

        # Check that default helpfulness criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            feedback_v2.Helpfulness.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_helpfulness_custom_criteria(self):
        """Test that helpfulness accepts custom criteria."""
        criteria_override = "Is the text technically accurate and detailed?"

        result = self.provider.helpfulness_with_cot_reasons(
            text="This is helpful text", criteria=criteria_override
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "Is the text technically accurate and detailed?",
            self.provider.last_system_prompt,
        )

    def test_maliciousness_custom_criteria(self):
        """Test that maliciousness accepts custom criteria."""
        criteria_override = "Does the text contain harmful security advice?"

        result = self.provider.maliciousness_with_cot_reasons(
            text="Some text to evaluate", criteria=criteria_override
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "Does the text contain harmful security advice?",
            self.provider.last_system_prompt,
        )

    def test_coherence_custom_criteria(self):
        """Test that coherence accepts custom criteria."""
        criteria_override = "Is the text logically structured?"

        result = self.provider.coherence_with_cot_reasons(
            text="Some text to evaluate", criteria=criteria_override
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "Is the text logically structured?",
            self.provider.last_system_prompt,
        )


class TestFeedbackCustomInstructions(TestCase):
    """Test that custom_instructions parameter works correctly."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_execution_efficiency_default_no_custom_instructions(self):
        """Test execution efficiency without custom instructions."""
        trace = "1. Step one\n2. Step two\n3. Step three"

        result = self.provider.execution_efficiency_with_cot_reasons(
            trace=trace
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], dict)

        # Check that prompt doesn't have custom instructions
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertEqual(
            feedback_v2.ExecutionEfficiency.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_execution_efficiency_with_custom_instructions(self):
        """Test execution efficiency with custom instructions."""
        trace = "1. Step one\n2. Step two\n3. Step three"
        custom_instructions = "CRITICAL: Ignore step 2 in the trace!"

        result = self.provider.execution_efficiency_with_cot_reasons(
            trace=trace, custom_instructions=custom_instructions
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "CRITICAL: Ignore step 2 in the trace!",
            self.provider.last_system_prompt,
        )

    def test_execution_efficiency_with_both_criteria_and_custom_instructions(
        self,
    ):
        """Test execution efficiency with both criteria and custom instructions."""
        trace = "1. Step one\n2. Step two\n3. Step three"
        criteria_override = "Judge how detailed the trace is."
        custom_instructions = "Focus on the level of detail in each step."

        result = self.provider.execution_efficiency_with_cot_reasons(
            trace=trace,
            criteria=criteria_override,
            custom_instructions=custom_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that both criteria and custom instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertNotIn(
            feedback_v2.ExecutionEfficiency.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertIn(
            "Judge how detailed the trace is.", self.provider.last_system_prompt
        )
        self.assertIn(
            "Focus on the level of detail in each step.",
            self.provider.last_system_prompt,
        )

    def test_plan_adherence_with_custom_instructions(self):
        """Test plan adherence with custom instructions."""
        trace = "Plan: Do X, Y, Z\nExecution: Did X, skipped Y, did Z"
        custom_instructions = "Skipping Y is acceptable in this context."

        result = self.provider.plan_adherence_with_cot_reasons(
            trace=trace, custom_instructions=custom_instructions
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "Skipping Y is acceptable in this context.",
            self.provider.last_system_prompt,
        )


class TestFeedbackIntegrationWithFeedbackClass(TestCase):
    """Test that Feedback class properly passes criteria and custom_instructions."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_feedback_with_criteria(self):
        """Test that Feedback class passes criteria to the implementation."""
        criteria_override = "Custom evaluation criteria"

        feedback = Feedback(
            self.provider.helpfulness_with_cot_reasons,
            name="Helpfulness",
            criteria=criteria_override,
        )

        # Check that criteria override is stored
        self.assertEqual(feedback.criteria, criteria_override)

    def test_feedback_with_custom_instructions(self):
        """Test that Feedback class passes custom_instructions to the implementation."""
        custom_instructions = "Special instructions for evaluation"

        feedback = Feedback(
            self.provider.execution_efficiency_with_cot_reasons,
            name="Execution Efficiency",
            custom_instructions=custom_instructions,
        )

        # Check that custom instructions are stored
        self.assertEqual(feedback.custom_instructions, custom_instructions)

    def test_feedback_with_both_criteria_and_custom_instructions(self):
        """Test that Feedback class handles both criteria and custom_instructions."""
        criteria_override = "Evaluate the efficiency"
        custom_instructions = "Focus on redundant steps"

        feedback = Feedback(
            self.provider.execution_efficiency_with_cot_reasons,
            name="Execution Efficiency",
            criteria=criteria_override,
            custom_instructions=custom_instructions,
        )

        # Check that criteria override and custom instructions are stored
        self.assertEqual(feedback.criteria, criteria_override)
        self.assertEqual(feedback.custom_instructions, custom_instructions)


class TestLegacyFeedbackFunctionsWithoutCustomInstructions(TestCase):
    """Test that legacy feedback functions work without custom_instructions parameter."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_legacy_functions_accept_only_criteria(self):
        """Test that legacy functions like helpfulness and maliciousness work with criteria only."""
        criteria_override = "Criteria override"

        # These should work without custom_instructions parameter
        legacy_functions = [
            self.provider.helpfulness_with_cot_reasons,
            self.provider.maliciousness_with_cot_reasons,
            self.provider.coherence_with_cot_reasons,
            self.provider.correctness_with_cot_reasons,
            self.provider.conciseness_with_cot_reasons,
        ]

        # Check that legacy functions work with only criteria
        for func in legacy_functions:
            feedback_func = Feedback(func, name=func.__name__)
            self.assertNotEqual(feedback_func.criteria, criteria_override)
            feedback_func_criteria_override = Feedback(
                func, name=func.__name__, criteria=criteria_override
            )
            self.assertEqual(
                feedback_func_criteria_override.criteria, criteria_override
            )


if __name__ == "__main__":
    unittest.main()
