"""Integration tests for feedback-function pipeline wiring."""

from typing import Dict, Optional, Tuple
import unittest

from trulens.core import Metric
from trulens.feedback import llm_provider


class MockLLMProvider(llm_provider.LLMProvider):
    """Mock provider that records prompts and returns deterministic scores."""

    model_config = {"extra": "allow"}

    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)

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
        return 2 / 3

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
        return 2 / 3, {"reason": "mock reason"}


class TestFeedbackPipelineWiring(unittest.TestCase):
    """Run major feedback functions through Metric + LLMProvider pipeline."""

    def setUp(self) -> None:
        self.provider = MockLLMProvider()

    def assert_pipeline_result(
        self,
        metric: Metric,
        expected_system_substrings: tuple[str, ...],
        expected_user_substrings: tuple[str, ...],
        **inputs,
    ) -> None:
        result = metric.run(**inputs)

        self.assertAlmostEqual(result.result, 2 / 3)
        self.assertGreater(len(self.provider.last_system_prompt or ""), 0)
        self.assertGreater(len(self.provider.last_user_prompt or ""), 0)

        for expected in expected_system_substrings:
            self.assertIn(expected, self.provider.last_system_prompt)
        for expected in expected_user_substrings:
            self.assertIn(expected, self.provider.last_user_prompt)

    def test_relevance_pipeline_wiring(self) -> None:
        metric = Metric(self.provider.relevance, name="Answer Relevance")

        self.assert_pipeline_result(
            metric,
            expected_system_substrings=(
                "RELEVANCE",
                "RESPONSE must be relevant",
            ),
            expected_user_substrings=(
                "What is TruLens?",
                "TruLens evaluates LLM applications.",
            ),
            prompt="What is TruLens?",
            response="TruLens evaluates LLM applications.",
        )

    def test_context_relevance_pipeline_wiring(self) -> None:
        metric = Metric(
            self.provider.context_relevance, name="Context Relevance"
        )

        self.assert_pipeline_result(
            metric,
            expected_system_substrings=("SEARCH RESULT", "USER QUERY"),
            expected_user_substrings=(
                "What is TruLens?",
                "TruLens evaluates LLM application quality.",
            ),
            question="What is TruLens?",
            context="TruLens evaluates LLM application quality.",
        )

    def test_groundedness_pipeline_wiring(self) -> None:
        metric = Metric(
            self.provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
        )

        self.assert_pipeline_result(
            metric,
            expected_system_substrings=(
                "directly supported by the source",
                "grounded",
            ),
            expected_user_substrings=(
                "TruLens evaluates LLM application quality.",
                "TruLens is used for LLM evaluation.",
            ),
            source="TruLens evaluates LLM application quality.",
            statement="TruLens is used for LLM evaluation.",
        )

    def test_sentiment_pipeline_wiring(self) -> None:
        metric = Metric(self.provider.sentiment, name="Sentiment")

        self.assert_pipeline_result(
            metric,
            expected_system_substrings=(
                "SENTIMENT grader",
                "Criteria for evaluating sentiment",
            ),
            expected_user_substrings=("This response is helpful and clear.",),
            text="This response is helpful and clear.",
        )


if __name__ == "__main__":
    unittest.main()
