import unittest

from trulens.core import Provider
from trulens.core.feedback import Feedback
from trulens.core.guardrails.base import block_input
from trulens.core.guardrails.base import block_output
from trulens.core.guardrails.base import context_filter


class DummyProvider(Provider):
    def dummy_feedback_low(self, query: str) -> float:
        """
        A dummy function to always return 0.2
        """
        return 0.2

    def dummy_feedback_high(self, query: str) -> float:
        """
        A dummy function to always return 0.8
        """
        return 0.8

    def dummy_context_relevance_low(self, query: str, context: str) -> float:
        """
        A dummy context relevance to always return 0.2
        """
        return 0.2

    def dummy_context_relevance_high(self, query: str, context: str) -> float:
        """
        A dummy context relevance to always return 0.8
        """
        return 0.8


dummy_provider = DummyProvider()

f_dummy_feedback_low = Feedback(dummy_provider.dummy_feedback_low)
f_dummy_feedback_high = Feedback(dummy_provider.dummy_feedback_high)
f_dummy_context_relevance_low = Feedback(
    dummy_provider.dummy_context_relevance_low
)
f_dummy_context_relevance_high = Feedback(
    dummy_provider.dummy_context_relevance_high
)


class TestGuardrailDecorators(unittest.TestCase):
    def test_context_filter(self):
        threshold = 0.5

        @context_filter(f_dummy_context_relevance_low, threshold, "query")
        def retrieve(query: str) -> list:
            return ["context1", "context2", "context3"]

        filtered_contexts = retrieve("example query")
        self.assertEqual(filtered_contexts, [])

    def test_no_context_filter(self):
        threshold = 0.5

        @context_filter(f_dummy_context_relevance_high, threshold, "query")
        def retrieve(query: str) -> list:
            return ["context1", "context2", "context3"]

        filtered_contexts = retrieve("example query")
        self.assertEqual(
            filtered_contexts, ["context1", "context2", "context3"]
        )

    def test_block_input(self):
        threshold = 0.5

        @block_input(f_dummy_feedback_low, threshold, "query")
        def generate_completion(query: str, context_str: list) -> str:
            return "Completion"

        result = generate_completion("example query", [])
        self.assertEqual(result, None)

    def test_no_block_input(self):
        threshold = 0.5

        @block_input(f_dummy_feedback_high, threshold, "query")
        def generate_completion(query: str, context_str: list) -> str:
            return "Completion"

        result = generate_completion("example query", [])
        self.assertEqual(result, "Completion")

    def test_block_output(self):
        threshold = 0.5

        @block_output(f_dummy_feedback_low, threshold)
        def chat(prompt: str) -> str:
            return "Response"

        result = chat("example prompt")
        self.assertEqual(result, None)

    def test_no_block_output(self):
        threshold = 0.5

        @block_output(f_dummy_feedback_high, threshold)
        def chat(prompt: str) -> str:
            return "Response"

        result = chat("example prompt")
        self.assertEqual(result, "Response")


if __name__ == "__main__":
    unittest.main()
