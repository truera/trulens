"""Integration tests for LLMProvider method-to-template wiring."""

from typing import Dict, Optional, Tuple
import unittest

from trulens.feedback import llm_provider
from trulens.feedback.dummy.endpoint import DummyEndpoint


class MockLLMProvider(llm_provider.LLMProvider):
    """Mock provider that records prompts assembled by LLMProvider methods."""

    model_config = {"extra": "allow"}

    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None

    def __init__(self, **kwargs):
        # groundedness_measure_with_cot_reasons() asserts self.endpoint is
        # not None before reaching the (overridden) scoring methods below,
        # unlike the other feedback functions covered by this test suite —
        # so a real Endpoint-like object is required here, not just None.
        # DummyEndpoint is trulens' own no-network-calls test double, used
        # the same way in trulens.feedback.dummy.provider.
        super().__init__(
            endpoint=DummyEndpoint(name="dummyendpoint"),
            model_engine="mock-model",
            **kwargs,
        )

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
        return 0.67

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
        return 0.67, {"reason": "mock reason"}


class TestMethodTemplateWiring(unittest.TestCase):
    """Verify major LLMProvider methods assemble prompts from the intended templates."""

    def setUp(self) -> None:
        self.provider = MockLLMProvider()

    def assert_prompt_wiring(
        self,
        system_substrings: tuple[str, ...],
        user_substrings: tuple[str, ...],
    ) -> None:
        system_prompt = self.provider.last_system_prompt or ""
        user_prompt = self.provider.last_user_prompt or ""

        self.assertGreater(len(system_prompt), 0)
        for expected in system_substrings:
            self.assertIn(expected, system_prompt)
        for expected in user_substrings:
            self.assertIn(expected, user_prompt)

    def test_relevance_uses_prompt_response_relevance_template(self) -> None:
        self.provider.relevance(
            prompt="What is TruLens?",
            response="TruLens evaluates LLM applications.",
        )

        self.assert_prompt_wiring(
            system_substrings=("RELEVANCE", "RESPONSE must be relevant"),
            user_substrings=(
                "What is TruLens?",
                "TruLens evaluates LLM applications.",
            ),
        )

    def test_context_relevance_uses_context_relevance_template(self) -> None:
        self.provider.context_relevance(
            question="What is TruLens?",
            context="TruLens evaluates LLM application quality.",
        )

        self.assert_prompt_wiring(
            system_substrings=("SEARCH RESULT", "USER QUERY"),
            user_substrings=(
                "What is TruLens?",
                "TruLens evaluates LLM application quality.",
            ),
        )

    def test_groundedness_uses_groundedness_template(self) -> None:
        self.provider.groundedness_measure_with_cot_reasons(
            source="TruLens evaluates LLM application quality.",
            statement="TruLens is used for LLM evaluation.",
        )

        self.assert_prompt_wiring(
            system_substrings=("directly supported by the source", "grounded"),
            user_substrings=(
                "TruLens evaluates LLM application quality.",
                "TruLens is used for LLM evaluation.",
            ),
        )

    def test_sentiment_uses_sentiment_template(self) -> None:
        self.provider.sentiment(text="This response is helpful and clear.")

        self.assert_prompt_wiring(
            system_substrings=(
                "SENTIMENT grader",
                "Criteria for evaluating sentiment",
            ),
            user_substrings=("This response is helpful and clear.",),
        )

    def test_stereotypes_uses_stereotypes_template(self) -> None:
        self.provider.stereotypes(
            prompt="Describe the engineer.",
            response="The engineer solved the problem carefully.",
        )

        self.assert_prompt_wiring(
            system_substrings=("gender or race", "PROMPT", "RESPONSE"),
            user_substrings=(
                "Describe the engineer.",
                "The engineer solved the problem carefully.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
