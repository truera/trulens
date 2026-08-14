# pyright: reportMissingImports=false, reportMissingModuleSource=false
import asyncio

import httpx
import openai
import pytest
from trulens.apps.app import TruApp
from trulens.apps.app import instrument
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

_SSE_BODY = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
    b"data: [DONE]\n\n"
)


def _mock_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_SSE_BODY,
    )


@pytest.mark.optional
class TestAsyncOpenAIStreamingSpanAttributes(OtelTestCase):
    """Exercises the *real* class-level `AsyncOpenAI.post` instrumentation
    (not an instance-level mock swap, which would bypass it) via a mocked
    httpx transport, so this runs without network access while still
    proving the fix that made this possible: `AsyncOpenAI.post` must
    piggyback on the caller's already-open span (`create_new_span=False`)
    rather than creating its own, which would always close before the
    caller starts consuming the stream.
    """

    def test_async_openai_streaming_records_span_attributes(self):
        oai_client = openai.AsyncOpenAI(
            api_key="test",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_mock_handler)
            ),
        )

        class App:
            @instrument
            async def stream_completion(self, prompt: str) -> str:
                completion = await oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    stream=True,
                    stream_options={"include_usage": True},
                    messages=[{"role": "user", "content": prompt}],
                )
                chunks = []
                async for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                return "".join(chunks)

        app = App()
        tru_app = TruApp(app, app_name="test_app", app_version="v1")

        async def run():
            with tru_app:
                return await app.stream_completion("hi")

        result = asyncio.run(run())
        self.assertEqual(result, "Hello world")

        events = self._get_events()
        found = False
        for attrs in events["record_attributes"]:
            if SpanAttributes.GENERATION.IS_STREAMING in attrs:
                found = True
                self.assertTrue(attrs[SpanAttributes.GENERATION.IS_STREAMING])
                self.assertEqual(
                    attrs[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 3
                )
                self.assertGreaterEqual(
                    attrs[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS], 0
                )
                self.assertGreater(
                    attrs[SpanAttributes.GENERATION.TOKENS_PER_SECOND], 0
                )
        self.assertTrue(found, "No span with IS_STREAMING found")
