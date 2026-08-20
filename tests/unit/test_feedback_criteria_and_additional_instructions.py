"""Unit tests for feedback function criteria and additional_instructions parameters."""

from typing import Dict, Optional, Tuple
import unittest
from unittest import TestCase
import warnings

from trulens.core import Metric
from trulens.feedback import llm_provider
from trulens.feedback.templates import agent as templates_agent
from trulens.feedback.templates import base as templates_base
from trulens.feedback.templates import conversation as templates_conversation
from trulens.feedback.templates import quality as templates_quality


class MockLLMProvider(llm_provider.LLMProvider):
    """Mock LLM provider for testing criteria and additional_instructions."""

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
            templates_quality.Helpfulness.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_helpfulness_custom_criteria(self):
        """Test that helpfulness accepts custom criteria without deprecation warning."""
        custom_criteria = "Is the text technically accurate and detailed?"

        # Using `criteria` parameter should NOT emit a deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.provider.helpfulness_with_cot_reasons(
                text="This is helpful text", criteria=custom_criteria
            )
            # Verify NO deprecation warning was issued for using `criteria`
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecation_warnings), 0)

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            custom_criteria,
            self.provider.last_system_prompt,
        )

    def test_maliciousness_custom_criteria(self):
        """Test that maliciousness accepts custom criteria."""
        custom_criteria = "Does the text contain harmful security advice?"

        result = self.provider.maliciousness_with_cot_reasons(
            text="Some text to evaluate", criteria=custom_criteria
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            custom_criteria,
            self.provider.last_system_prompt,
        )

    def test_coherence_custom_criteria(self):
        """Test that coherence accepts custom criteria."""
        custom_criteria = "Is the text logically structured?"

        result = self.provider.coherence_with_cot_reasons(
            text="Some text to evaluate", criteria=custom_criteria
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        # Check that custom criteria is in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            custom_criteria,
            self.provider.last_system_prompt,
        )


class TestFeedbackAdditionalInstructions(TestCase):
    """Test that additional_instructions parameter works correctly."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_execution_efficiency_default_no_additional_instructions(self):
        """Test execution efficiency without additional instructions."""
        trace = "1. Step one\n2. Step two\n3. Step three"

        result = self.provider.execution_efficiency_with_cot_reasons(
            trace=trace
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], dict)

        # Check that prompt doesn't have additional instructions
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertEqual(
            templates_agent.ExecutionEfficiency.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_execution_efficiency_with_additional_instructions(self):
        """Test execution efficiency with additional instructions (new parameter name)."""
        trace = "1. Step one\n2. Step two\n3. Step three"
        additional_instructions = "CRITICAL: Ignore step 2 in the trace!"

        # Using the new `additional_instructions` parameter should NOT emit a deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.provider.execution_efficiency_with_cot_reasons(
                trace=trace, additional_instructions=additional_instructions
            )
            # Verify NO deprecation warning was issued for using the correct parameter name
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecation_warnings), 0)

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_execution_efficiency_with_deprecated_custom_instructions(self):
        """Test that deprecated custom_instructions parameter still works."""
        trace = "1. Step one\n2. Step two\n3. Step three"
        custom_instructions = "CRITICAL: Ignore step 2 in the trace!"

        # Using deprecated `custom_instructions` parameter should emit DeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.provider.execution_efficiency_with_cot_reasons(
                trace=trace, custom_instructions=custom_instructions
            )
            # Verify deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecation_warnings), 1)
            self.assertIn(
                "custom_instructions", str(deprecation_warnings[0].message)
            )
            self.assertIn(
                "additional_instructions", str(deprecation_warnings[0].message)
            )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that the instructions still reached the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            custom_instructions,
            self.provider.last_system_prompt,
        )

    def test_execution_efficiency_with_both_criteria_and_additional_instructions(
        self,
    ):
        """Test execution efficiency with both criteria and additional instructions."""
        trace = "1. Step one\n2. Step two\n3. Step three"
        custom_criteria = "Judge how detailed the trace is."
        additional_instructions = "Focus on the level of detail in each step."

        result = self.provider.execution_efficiency_with_cot_reasons(
            trace=trace,
            criteria=custom_criteria,
            additional_instructions=additional_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that both criteria and additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertNotIn(
            templates_agent.ExecutionEfficiency.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertIn(custom_criteria, self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_plan_adherence_with_additional_instructions(self):
        """Test plan adherence with additional instructions."""
        trace = "Plan: Do X, Y, Z\nExecution: Did X, skipped Y, did Z"
        additional_instructions = "Skipping Y is acceptable in this context."

        result = self.provider.plan_adherence_with_cot_reasons(
            trace=trace, additional_instructions=additional_instructions
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_correctness_with_additional_instructions(self):
        """Test correctness with additional instructions."""
        text = "The capital of France is Paris."
        additional_instructions = "Consider geographical accuracy only."

        result = self.provider.correctness_with_cot_reasons(
            text=text, additional_instructions=additional_instructions
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_conciseness_with_additional_instructions(self):
        """Test conciseness with additional instructions."""
        text = "This is a brief response."
        additional_instructions = "Technical jargon is acceptable if concise."

        result = self.provider.conciseness_with_cot_reasons(
            text=text, additional_instructions=additional_instructions
        )

        # Check that result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )


class TestConversationAdditionalInstructions(TestCase):
    """Test that conversation-level metrics accept additional_instructions."""

    def setUp(self):
        self.provider = MockLLMProvider()
        self.records = [
            {"input": "What is 2+2?", "output": "Four."},
            {"input": "Why?", "output": "Two pairs total four items."},
        ]

    def test_coherence_across_turns_default_no_additional_instructions(self):
        """Test coherence across turns without additional instructions."""
        result = self.provider.coherence_across_turns(records=self.records)

        # Check that result is valid
        self.assertIsInstance(result, float)

        # Check that the default system prompt is left untouched
        self.assertEqual(
            templates_conversation.CoherenceAcrossTurns.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_coherence_across_turns_with_additional_instructions(self):
        """Test coherence across turns with additional instructions."""
        additional_instructions = (
            "Assistant turns are structured SQL result rows, not prose."
        )

        result = self.provider.coherence_across_turns(
            records=self.records,
            additional_instructions=additional_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )
        self.assertIn(
            templates_conversation.CoherenceAcrossTurns.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_conversation_helpfulness_with_additional_instructions(self):
        """Test conversation helpfulness with additional instructions."""
        additional_instructions = "Terse answers are acceptable here."

        result = self.provider.conversation_helpfulness(
            records=self.records,
            additional_instructions=additional_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, float)

        # Check that additional instructions are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_topic_adherence_with_additional_instructions(self):
        """Test topic adherence with additional instructions."""
        additional_instructions = "Arithmetic follow-ups count as on topic."

        result = self.provider.topic_adherence(
            records=self.records,
            reference_topics=["arithmetic"],
            additional_instructions=additional_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, float)

        # Check that both the reference topics and the additional instructions
        # are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn("arithmetic", self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_agent_goal_accuracy_with_additional_instructions(self):
        """Test agent goal accuracy with additional instructions."""
        additional_instructions = "A partial answer does not achieve the goal."

        result = self.provider.agent_goal_accuracy(
            records=self.records,
            reference_goal="Explain why 2+2 is four.",
            additional_instructions=additional_instructions,
        )

        # Check that result is valid
        self.assertIsInstance(result, float)

        # Check that both the reference goal and the additional instructions
        # are in the prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            "Explain why 2+2 is four.", self.provider.last_system_prompt
        )
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_conversation_metric_through_metric_class(self):
        """Test that Metric passes additional_instructions to a conversation metric."""
        additional_instructions = "UNIQUE_CONVERSATION_INSTRUCTIONS_E2E_TEST"

        feedback = Metric(
            implementation=self.provider.coherence_across_turns,
            name="Coherence Across Turns",
            additional_instructions=additional_instructions,
        )

        result = feedback(records=self.records)

        # Verify result is valid
        self.assertIsInstance(result, float)

        # Verify additional_instructions reached the provider's prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(additional_instructions, self.provider.last_system_prompt)


class TestConversationCotReasons(TestCase):
    """Test the chain of thought variants of conversation-level metrics."""

    def setUp(self):
        self.provider = MockLLMProvider()
        self.records = [
            {"input": "What is 2+2?", "output": "Four."},
            {"input": "Why?", "output": "Two pairs total four items."},
        ]

    def _assert_cot_result(self, result):
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], dict)

    def test_coherence_across_turns_with_cot_reasons(self):
        """Test that coherence across turns emits reasons and the transcript."""
        additional_instructions = "Structured rows are coherent by convention."

        result = self.provider.coherence_across_turns_with_cot_reasons(
            records=self.records,
            additional_instructions=additional_instructions,
        )

        self._assert_cot_result(result)

        # Check that the reasons template and the transcript are in the user prompt
        self.assertIsNotNone(self.provider.last_user_prompt)
        self.assertIn(
            templates_base.COT_REASONS_TEMPLATE,
            self.provider.last_user_prompt,
        )
        self.assertIn(
            "Turn 1 User: What is 2+2?", self.provider.last_user_prompt
        )

        # Check that additional instructions are in the system prompt
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_conversation_helpfulness_with_cot_reasons(self):
        """Test that conversation helpfulness emits reasons."""
        additional_instructions = "Terse answers are acceptable here."

        result = self.provider.conversation_helpfulness_with_cot_reasons(
            records=self.records,
            additional_instructions=additional_instructions,
        )

        self._assert_cot_result(result)
        self.assertIn(
            templates_base.COT_REASONS_TEMPLATE,
            self.provider.last_user_prompt,
        )
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_topic_adherence_with_cot_reasons(self):
        """Test that topic adherence emits reasons."""
        additional_instructions = "Arithmetic follow-ups count as on topic."

        result = self.provider.topic_adherence_with_cot_reasons(
            records=self.records,
            reference_topics=["arithmetic"],
            additional_instructions=additional_instructions,
        )

        self._assert_cot_result(result)
        self.assertIn(
            templates_base.COT_REASONS_TEMPLATE,
            self.provider.last_user_prompt,
        )
        self.assertIn("arithmetic", self.provider.last_system_prompt)
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_agent_goal_accuracy_with_cot_reasons(self):
        """Test that agent goal accuracy emits reasons."""
        additional_instructions = "A partial answer does not achieve the goal."

        result = self.provider.agent_goal_accuracy_with_cot_reasons(
            records=self.records,
            reference_goal="Explain why 2+2 is four.",
            additional_instructions=additional_instructions,
        )

        self._assert_cot_result(result)
        self.assertIn(
            templates_base.COT_REASONS_TEMPLATE,
            self.provider.last_user_prompt,
        )
        self.assertIn(
            "Explain why 2+2 is four.", self.provider.last_system_prompt
        )
        self.assertIn(
            additional_instructions,
            self.provider.last_system_prompt,
        )

    def test_cot_variants_accept_a_transcript_string(self):
        """Test that the chain of thought variants accept a plain transcript."""
        transcript = "Turn 1 User: Hello\nTurn 1 Assistant: Hi"

        result = self.provider.coherence_across_turns_with_cot_reasons(
            records=transcript
        )

        self._assert_cot_result(result)
        self.assertIn(transcript, self.provider.last_user_prompt)


class TestFeedbackIntegrationWithFeedbackClass(TestCase):
    """Test that Feedback class properly passes criteria and additional_instructions."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_feedback_with_criteria(self):
        """Test that Feedback class passes criteria to the implementation."""
        custom_criteria = "Custom Criteria Test"

        feedback = Metric(
            implementation=self.provider.helpfulness_with_cot_reasons,
            name="Helpfulness",
            criteria=custom_criteria,
        )

        # Check that criteria is stored
        self.assertEqual(feedback.criteria, custom_criteria)

        # Actually call the feedback function end-to-end
        result = feedback(text="Some test text")

        # Verify result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Verify criteria reached the provider's prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(custom_criteria, self.provider.last_system_prompt)

    def test_feedback_with_additional_instructions(self):
        """Test that Feedback class passes additional_instructions to the implementation."""
        additional_instructions = "UNIQUE_ADDITIONAL_INSTRUCTIONS_E2E_TEST"

        feedback = Metric(
            implementation=self.provider.helpfulness_with_cot_reasons,
            name="Helpfulness",
            additional_instructions=additional_instructions,
        )

        # Check that additional instructions are stored
        self.assertEqual(
            feedback.additional_instructions, additional_instructions
        )

        # Actually call the feedback function end-to-end
        result = feedback(text="Some test text")

        # Verify result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Verify additional_instructions reached the provider's prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(additional_instructions, self.provider.last_system_prompt)

    def test_feedback_with_both_criteria_and_additional_instructions(self):
        """Test that Feedback class handles both criteria and additional_instructions."""
        custom_criteria = "UNIQUE_CRITERIA_COMBO_E2E_TEST"
        additional_instructions = "UNIQUE_INSTRUCTIONS_COMBO_E2E_TEST"

        feedback = Metric(
            implementation=self.provider.helpfulness_with_cot_reasons,
            name="Helpfulness",
            criteria=custom_criteria,
            additional_instructions=additional_instructions,
        )

        # Check that criteria and additional instructions are stored
        self.assertEqual(feedback.criteria, custom_criteria)
        self.assertEqual(
            feedback.additional_instructions, additional_instructions
        )

        # Actually call the feedback function end-to-end
        result = feedback(text="Some test text")

        # Verify result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Verify both criteria and additional_instructions reached the provider's prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(custom_criteria, self.provider.last_system_prompt)
        self.assertIn(additional_instructions, self.provider.last_system_prompt)

    def test_feedback_with_deprecated_custom_instructions(self):
        """Test that deprecated custom_instructions parameter still works end-to-end."""
        custom_instructions = "UNIQUE_DEPRECATED_CUSTOM_INSTRUCTIONS_E2E_TEST"

        # Using deprecated custom_instructions should emit a warning but still work
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            feedback = Metric(
                implementation=self.provider.helpfulness_with_cot_reasons,
                name="Helpfulness",
                custom_instructions=custom_instructions,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            custom_instructions_warnings = [
                x
                for x in deprecation_warnings
                if "custom_instructions" in str(x.message)
            ]
            self.assertEqual(len(custom_instructions_warnings), 1)
            self.assertIn(
                "additional_instructions",
                str(custom_instructions_warnings[0].message),
            )

        # Check that value is stored (under additional_instructions now)
        self.assertEqual(feedback.additional_instructions, custom_instructions)

        # Actually call the feedback function end-to-end
        result = feedback(text="Some test text")

        # Verify result is valid
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)

        # Verify the value reached the provider's prompt
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(custom_instructions, self.provider.last_system_prompt)

    def test_simple_feedback_with_both_criteria_and_additional_instructions(
        self,
    ):
        """Test that simple feedback with both criteria and additional instructions works."""
        criteria = "Custom Criteria Test"
        additional_instructions = "Additional Instructions Test"

        feedback = Metric(
            implementation=self.provider.conciseness_with_cot_reasons,
            name="Conciseness",
            criteria=criteria,
            additional_instructions=additional_instructions,
        )

        result = feedback(text="Some test text")
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(criteria, self.provider.last_system_prompt)
        self.assertIn(additional_instructions, self.provider.last_system_prompt)


if __name__ == "__main__":
    unittest.main()
