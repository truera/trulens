"""Tests for streaming instrumentation: TTFT, throughput and chunk counts."""

import asyncio
import os
import time
import unittest
from unittest import mock

import openai
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from opentelemetry import trace
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from trulens.core.otel.instrument import instrument
from trulens.core.otel.recording import Recording
from trulens.experimental.otel_tracing.core.session import (
    _set_up_tracer_provider,
)
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai.endpoint import OpenAICostComputer

_MODEL = "gpt-3.5-turbo"
"""A model langchain knows the price of, so cost assertions are meaningful."""

_FIRST_TOKEN_DELAY = 0.05
"""Long enough that a time-to-first-token measurement cannot round to zero."""


def _content_chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model=_MODEL,
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=None,
            )
        ],
    )


def _stop_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model=_MODEL,
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
    )


def _usage_chunk(
    prompt_tokens: int, completion_tokens: int
) -> ChatCompletionChunk:
    """The final chunk sent when `stream_options={"include_usage": True}`."""
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model=_MODEL,
        object="chat.completion.chunk",
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_stream(chunks, delay: float = 0.0) -> openai.Stream:
    """Build an openai Stream over `chunks` without touching the network.

    `__init__` is bypassed deliberately: it wants a live client and an httpx
    response, and everything under test reads only `_iterator`.
    """

    def iterator():
        for i, chunk in enumerate(chunks):
            if i == 0 and delay:
                time.sleep(delay)
            yield chunk

    stream = openai.Stream.__new__(openai.Stream)
    stream._iterator = iterator()
    return stream


class TestOpenAIStreamingCostTracking(unittest.TestCase):
    """The openai provider path, which knows token counts and chunk timing."""

    def setUp(self) -> None:
        # Cost tracking builds an OpenAIEndpoint, whose client insists on a key
        # even though nothing here reaches the network.
        patcher = mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test-not-a-real-key"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        trace.get_tracer_provider().add_span_processor(self.span_processor)
        self.tracer = trace.get_tracer_provider().get_tracer(__name__)
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        return super().tearDown()

    def _consume(self, chunks, delay: float = 0.0, stop_after: int = -1):
        """Run a stream through cost tracking and return the finished span."""
        stream = _make_stream(chunks, delay=delay)
        with self.tracer.start_as_current_span("llm_call"):
            returned = OpenAICostComputer.handle_response(stream)
            for i, _ in enumerate(stream):
                if i == stop_after:
                    break

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        return returned, spans[0]

    def test_streaming_is_flagged_immediately(self) -> None:
        """`is_streaming` is known before a single chunk has been consumed."""
        returned, _ = self._consume([_content_chunk("hi"), _stop_chunk()])
        self.assertTrue(returned[SpanAttributes.GENERATION.IS_STREAMING])

    def test_non_streaming_is_flagged(self) -> None:
        response = openai.types.chat.ChatCompletion(
            id="chatcmpl-test",
            created=0,
            model=_MODEL,
            object="chat.completion",
            choices=[],
            usage=CompletionUsage(
                prompt_tokens=3, completion_tokens=4, total_tokens=7
            ),
        )
        with self.tracer.start_as_current_span("llm_call"):
            returned = OpenAICostComputer.handle_response(response)

        self.assertFalse(returned[SpanAttributes.GENERATION.IS_STREAMING])

    def test_embedding_response_is_not_called_a_generation(self) -> None:
        """Cost tracking wraps every `create`, not just generation ones."""
        response = openai.types.CreateEmbeddingResponse(
            data=[],
            model="text-embedding-3-small",
            object="list",
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=5, total_tokens=5
            ),
        )
        with self.tracer.start_as_current_span("embedding_call"):
            returned = OpenAICostComputer.handle_response(response)

        self.assertNotIn(SpanAttributes.GENERATION.IS_STREAMING, returned)

    def test_chunks_and_ttft_recorded(self) -> None:
        chunks = [_content_chunk(c) for c in "abcd"] + [_stop_chunk()]
        _, span = self._consume(chunks, delay=_FIRST_TOKEN_DELAY)

        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED],
            len(chunks),
        )
        # The first chunk is pulled eagerly by cost tracking, so the measured
        # time-to-first-token must reflect the wait for it.
        self.assertGreaterEqual(
            span.attributes[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS],
            _FIRST_TOKEN_DELAY * 1000.0 * 0.5,
        )

    def test_usage_chunk_populates_tokens_and_cost(self) -> None:
        chunks = [
            _content_chunk("a"),
            _content_chunk("b"),
            _stop_chunk(),
            _usage_chunk(prompt_tokens=11, completion_tokens=7),
        ]
        _, span = self._consume(chunks)

        self.assertEqual(
            span.attributes[SpanAttributes.COST.NUM_PROMPT_TOKENS], 11
        )
        self.assertEqual(
            span.attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 7
        )
        self.assertEqual(span.attributes[SpanAttributes.COST.NUM_TOKENS], 18)
        self.assertGreater(span.attributes[SpanAttributes.COST.COST], 0.0)
        self.assertEqual(span.attributes[SpanAttributes.COST.MODEL], _MODEL)

    def test_tokens_per_second_requires_known_token_counts(self) -> None:
        """Throughput is reported only when usage says how many tokens there were."""
        without_usage = [_content_chunk("a"), _stop_chunk()]
        _, span = self._consume(without_usage)
        self.assertNotIn(
            SpanAttributes.GENERATION.TOKENS_PER_SECOND, span.attributes
        )

        self.setUp()  # fresh exporter for the second stream
        with_usage = [
            _content_chunk("a"),
            _content_chunk("b"),
            _stop_chunk(),
            _usage_chunk(prompt_tokens=11, completion_tokens=7),
        ]
        _, span = self._consume(with_usage, delay=_FIRST_TOKEN_DELAY)
        self.assertGreater(
            span.attributes[SpanAttributes.GENERATION.TOKENS_PER_SECOND], 0.0
        )

    def test_abandoned_stream_records_what_it_produced(self) -> None:
        """A stream dropped part way through still reports its chunk count."""
        chunks = [_content_chunk(c) for c in "abcdefgh"] + [_stop_chunk()]
        _, span = self._consume(chunks, stop_after=2)

        chunks_received = span.attributes[
            SpanAttributes.GENERATION.CHUNKS_RECEIVED
        ]
        self.assertEqual(chunks_received, 3)
        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])

    def test_async_stream_is_measured(self) -> None:
        chunks = [_content_chunk("a"), _content_chunk("b"), _stop_chunk()]

        async def iterator():
            for chunk in chunks:
                yield chunk

        stream = openai.AsyncStream.__new__(openai.AsyncStream)
        stream._iterator = iterator()

        async def run():
            with self.tracer.start_as_current_span("llm_call"):
                OpenAICostComputer.handle_response(stream)
                async for _ in stream:
                    pass

        asyncio.run(run())

        span = self.exporter.get_finished_spans()[0]
        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED],
            len(chunks),
        )


class TestGenericStreamingInstrumentation(unittest.TestCase):
    """The framework-agnostic path: any instrumented generator."""

    @classmethod
    def setUpClass(cls) -> None:
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        trace.get_tracer_provider().add_span_processor(self.span_processor)
        self.tokens = []
        self.tokens.append(
            context_api.attach(
                set_baggage("__trulens_recording__", Recording(None))
            )
        )
        self.tokens.append(
            context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, "test_record_id")
            )
        )
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        remove_baggage("__trulens_recording__")
        remove_baggage(SpanAttributes.RECORD_ID)
        for token in self.tokens[::-1]:
            context_api.detach(token)
        return super().tearDown()

    def test_sync_generator_records_streaming_attributes(self) -> None:
        @instrument()
        def my_function():
            time.sleep(_FIRST_TOKEN_DELAY)
            yield "Kojikun"
            yield "Nolan"
            yield "Sachiboy"

        for _ in my_function():
            pass

        span = self.exporter.get_finished_spans()[0]
        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 3
        )
        self.assertGreaterEqual(
            span.attributes[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS],
            _FIRST_TOKEN_DELAY * 1000.0 * 0.5,
        )
        # Token counts are unknowable here, so throughput must not be guessed.
        self.assertNotIn(
            SpanAttributes.GENERATION.TOKENS_PER_SECOND, span.attributes
        )

    def test_non_generator_is_not_marked_as_streaming(self) -> None:
        @instrument()
        def my_function():
            return "Kojikun"

        my_function()

        span = self.exporter.get_finished_spans()[0]
        self.assertNotIn(
            SpanAttributes.GENERATION.IS_STREAMING, span.attributes
        )

    def test_non_streaming_provider_call_does_not_suppress_measurement(
        self,
    ) -> None:
        """A provider reporting no stream must not silence the generator's own.

        Only a provider that actually streamed knows better than we do; one
        that made an ordinary call says nothing about what the generator
        wrapped around it is doing.
        """

        @instrument()
        def my_function():
            # What cost tracking writes for a non-streamed provider call.
            trace.get_current_span().set_attribute(
                SpanAttributes.GENERATION.IS_STREAMING, False
            )
            yield "Kojikun"
            yield "Nolan"

        for _ in my_function():
            pass

        span = self.exporter.get_finished_spans()[0]
        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 2
        )

    def test_provider_measurements_survive_the_wrapping_generator(
        self,
    ) -> None:
        """The openai numbers must win over the generator's coarser view.

        A streaming app yields only the chunks carrying content, so counting
        yields would undercount, and it can say nothing about tokens per
        second. Whatever the provider measured has to reach the span intact.
        """
        chunks = [
            _content_chunk("a"),
            _content_chunk("b"),
            _stop_chunk(),
            _usage_chunk(prompt_tokens=11, completion_tokens=7),
        ]
        stream = _make_stream(chunks, delay=_FIRST_TOKEN_DELAY)

        @instrument()
        def stream_completion():
            OpenAICostComputer.handle_response(stream)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test-not-a-real-key"}
        ):
            yielded = list(stream_completion())

        self.assertEqual(yielded, ["a", "b"])

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED],
            len(chunks),
        )
        self.assertGreater(
            span.attributes[SpanAttributes.GENERATION.TOKENS_PER_SECOND], 0.0
        )
        self.assertEqual(
            span.attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 7
        )

    def test_async_generator_records_streaming_attributes(self) -> None:
        @instrument()
        async def my_function():
            await asyncio.sleep(_FIRST_TOKEN_DELAY)
            yield "Kojikun"
            yield "Nolan"

        async def run():
            async for _ in my_function():
                pass

        asyncio.run(run())

        span = self.exporter.get_finished_spans()[0]
        self.assertTrue(span.attributes[SpanAttributes.GENERATION.IS_STREAMING])
        self.assertEqual(
            span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 2
        )
        self.assertGreaterEqual(
            span.attributes[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS],
            _FIRST_TOKEN_DELAY * 1000.0 * 0.5,
        )


if __name__ == "__main__":
    unittest.main()
