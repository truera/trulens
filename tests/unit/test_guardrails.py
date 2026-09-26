import unittest

from trulens.core import Metric
from trulens.core import Provider
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

f_dummy_feedback_low = Metric(implementation=dummy_provider.dummy_feedback_low)
f_dummy_feedback_high = Metric(
    implementation=dummy_provider.dummy_feedback_high
)
f_dummy_context_relevance_low = Metric(
    implementation=dummy_provider.dummy_context_relevance_low
)
f_dummy_context_relevance_high = Metric(
    implementation=dummy_provider.dummy_context_relevance_high
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
            set(filtered_contexts), set(["context1", "context2", "context3"])
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

    def test_block_output_streaming_cuts_off_after_checkpoint(self):
        """Once the checkpoint check trips, no further chunks should be
        yielded (only the fallback, if one is configured); chunks already
        yielded before that point are (necessarily) not retracted."""
        threshold = 0.5

        @block_output(
            f_dummy_feedback_low,
            threshold,
            return_value="[blocked]",
            check_every_n_chunks=2,
        )
        def chat(prompt: str):
            for word in ["one", "two", "three", "four", "five", "six"]:
                yield word

        result = list(chat("example prompt"))
        # Checked after 2 chunks ("one", "two"); low feedback trips the
        # threshold immediately, so those two plus the fallback is all we
        # should see.
        self.assertEqual(result, ["one", "two", "[blocked]"])

    def test_block_output_streaming_no_fallback_just_stops(self):
        threshold = 0.5

        @block_output(f_dummy_feedback_low, threshold, check_every_n_chunks=1)
        def chat(prompt: str):
            for word in ["one", "two", "three"]:
                yield word

        result = list(chat("example prompt"))
        self.assertEqual(result, ["one"])

    def test_no_block_output_streaming_yields_everything(self):
        threshold = 0.5

        @block_output(f_dummy_feedback_high, threshold, check_every_n_chunks=2)
        def chat(prompt: str):
            for word in ["one", "two", "three", "four"]:
                yield word

        result = list(chat("example prompt"))
        self.assertEqual(result, ["one", "two", "three", "four"])

    def test_block_output_streaming_final_check_on_short_tail(self):
        """A stream shorter than check_every_n_chunks never hits a
        checkpoint mid-stream, but should still get a final check (for
        observability -- it can no longer block anything by then, so all
        chunks are still yielded)."""
        threshold = 0.5

        @block_output(f_dummy_feedback_low, threshold, check_every_n_chunks=10)
        def chat(prompt: str):
            for word in ["one", "two"]:
                yield word

        result = list(chat("example prompt"))
        self.assertEqual(result, ["one", "two"])

    def test_block_output_async_streaming_cuts_off_after_checkpoint(self):
        import asyncio

        threshold = 0.5

        @block_output(
            f_dummy_feedback_low,
            threshold,
            return_value="[blocked]",
            check_every_n_chunks=2,
        )
        async def chat(prompt: str):
            for word in ["one", "two", "three", "four"]:
                yield word

        async def run():
            return [chunk async for chunk in chat("example prompt")]

        result = asyncio.run(run())
        self.assertEqual(result, ["one", "two", "[blocked]"])
