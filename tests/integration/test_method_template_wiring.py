"""Integration tests verifying method-to-template wiring in LLMProvider.

Each test asserts that the provider method assembles a system prompt whose
content comes from the correct template class — catching copy-paste bugs where,
e.g., relevance() accidentally uses the ContextRelevance template.
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
        return 2 / max_score_val, {"reason": "test"}


class TestMethodTemplateWiring(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_relevance_uses_prompt_response_relevance_template(self):
        self.provider.relevance(
            prompt="What is TruLens?",
            response="TruLens is an evaluation framework.",
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_rag.PromptResponseRelevance.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertNotIn(
            templates_rag.ContextRelevance.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_context_relevance_uses_context_relevance_template(self):
        self.provider.context_relevance(
            question="What is TruLens?",
            context="TruLens evaluates LLM applications.",
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_rag.ContextRelevance.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertNotIn(
            templates_rag.PromptResponseRelevance.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_sentiment_uses_sentiment_template(self):
        self.provider.sentiment(text="This is excellent!")
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_quality.Sentiment.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertNotIn(
            templates_rag.ContextRelevance.system_prompt,
            self.provider.last_system_prompt,
        )

    def test_groundedness_uses_groundedness_template(self):
        self.provider.groundedness_measure_with_cot_reasons(
            source="TruLens was created by TruEra.",
            statement="TruLens was created by TruEra.",
        )
        self.assertIsNotNone(self.provider.last_system_prompt)
        self.assertIn(
            templates_rag.Groundedness.system_prompt,
            self.provider.last_system_prompt,
        )
        self.assertNotIn(
            templates_rag.ContextRelevance.system_prompt,
            self.provider.last_system_prompt,
        )


if __name__ == "__main__":
    unittest.main()
