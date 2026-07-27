"""Integration tests for the feedback function pipeline.

Verifies the full pipeline: provider method call → template rendering →
prompt assembly → generate_score call → float returned.

Each test uses a MockLLMProvider that intercepts generate_score /
generate_score_and_reasons so the real prompt-assembly path executes without
a live LLM endpoint.
"""

from typing import ClassVar, Dict, Optional, Tuple
import unittest

from trulens.core.feedback import endpoint as core_endpoint
from trulens.feedback import llm_provider
from trulens.feedback.templates import quality as templates_quality
from trulens.feedback.templates import rag as templates_rag


class MockLLMProvider(llm_provider.LLMProvider):
    model_config: ClassVar[dict] = {"extra": "allow"}

    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(
            endpoint=core_endpoint.Endpoint(name="mock-endpoint"),
            model_engine="mock-model",
            **kwargs,
        )

    def _create_chat_completion(self, **kwargs) -> str:
        return "Score: 2\nReason: test"

    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 2 / max_score_val

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 2 / max_score_val, {"reason": "test reason"}


class TestFeedbackPipelineWiring(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_relevance_returns_float_in_range(self):
        result = self.provider.relevance(
            prompt="What is TruLens?",
            response="TruLens is an evaluation framework.",
        )
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_relevance_assembles_correct_template(self):
        self.provider.relevance(
            prompt="What is TruLens?",
            response="TruLens is an evaluation framework.",
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_rag.PromptResponseRelevance.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_relevance_user_prompt_contains_inputs(self):
        self.provider.relevance(
            prompt="What is TruLens?",
            response="TruLens is an evaluation framework.",
        )
        self.assertIsNotNone(self.provider.last_user_prompt)
        self.assertIn("What is TruLens?", self.provider.last_user_prompt)
        self.assertIn(
            "TruLens is an evaluation framework.",
            self.provider.last_user_prompt,
        )

    def test_context_relevance_returns_float_in_range(self):
        result = self.provider.context_relevance(
            question="What is TruLens?",
            context="TruLens evaluates LLM applications.",
        )
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_context_relevance_assembles_correct_template(self):
        self.provider.context_relevance(
            question="What is TruLens?",
            context="TruLens evaluates LLM applications.",
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_rag.ContextRelevance.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_sentiment_returns_float_in_range(self):
        result = self.provider.sentiment(text="This product is excellent!")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_sentiment_assembles_correct_template(self):
        self.provider.sentiment(text="This product is excellent!")
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_quality.Sentiment.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_groundedness_returns_float_and_reasons(self):
        result = self.provider.groundedness_measure_with_cot_reasons(
            source="TruLens was created by TruEra.",
            statement="TruLens was created by TruEra.",
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        score, reasons = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(reasons, dict)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
