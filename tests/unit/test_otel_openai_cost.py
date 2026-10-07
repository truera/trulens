"""Tests that an openai call inside a recording is costed exactly once.

Async openai cost is recorded on `AsyncOpenAI.post`. Its resource methods,
such as `AsyncCompletions.create`, are `async def`s behind a sync decorator, so
cost tracking must not wrap them as well or the call is counted twice.

No network: the clients answer from an `httpx.MockTransport`.
"""

import asyncio
import os
from unittest import mock

import httpx
import pytest

pytest.importorskip("openai")
pytest.importorskip("trulens.providers.openai")

import openai
from trulens.apps import app as trulens_app
from trulens.core import session as core_session
from trulens.core.otel import instrument as core_instrument
from trulens.otel.semconv import trace as semconv_trace

import tests.util.otel_test_case

_SpanAttributes = semconv_trace.SpanAttributes

_PROMPT_TOKENS = 100
_COMPLETION_TOKENS = 50
_TOTAL_TOKENS = _PROMPT_TOKENS + _COMPLETION_TOKENS

_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Kojikun"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": _PROMPT_TOKENS,
        "completion_tokens": _COMPLETION_TOKENS,
        "total_tokens": _TOTAL_TOKENS,
    },
}

_MESSAGES = [{"role": "user", "content": "Who is the best baby?"}]


def _respond(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_COMPLETION)


class _AsyncApp:
    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_respond)
            ),
        )

    @core_instrument.instrument()
    async def query(self, question: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini", messages=_MESSAGES
        )
        return response.choices[0].message.content


class _SyncApp:
    def __init__(self) -> None:
        self.client = openai.OpenAI(
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(_respond)),
        )

    @core_instrument.instrument()
    def query(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=_MESSAGES
        )
        return response.choices[0].message.content


@pytest.mark.optional
class TestOtelOpenAICost(tests.util.otel_test_case.OtelTestCase):
    def setUp(self) -> None:
        # Cost tracking builds an OpenAIEndpoint, whose client insists on a key
        # even though nothing here reaches the network.
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        return super().setUp()

    def _costed_spans(self) -> list:
        """The `record_attributes` of every span that carries tokens."""
        core_session.TruSession().force_flush()
        costed = []
        for attributes in self._get_events()["record_attributes"]:
            if isinstance(attributes, str):
                attributes = eval(attributes)
            if attributes.get(_SpanAttributes.COST.NUM_TOKENS):
                costed.append(attributes)
        return costed

    def _assert_costed_once(self) -> None:
        costed = self._costed_spans()
        self.assertEqual(len(costed), 1)
        self.assertEqual(
            sum(a[_SpanAttributes.COST.NUM_TOKENS] for a in costed),
            _TOTAL_TOKENS,
        )
        self.assertEqual(
            costed[0][_SpanAttributes.COST.NUM_PROMPT_TOKENS], _PROMPT_TOKENS
        )
        self.assertGreater(costed[0][_SpanAttributes.COST.COST], 0)

    def test_async_chat_completion_is_costed_once(self) -> None:
        app = _AsyncApp()
        tru_app = trulens_app.TruApp(
            app, app_name="async_app", app_version="v1", main_method=app.query
        )
        with tru_app:
            self.assertEqual(asyncio.run(app.query("q")), "Kojikun")

        self._assert_costed_once()

    def test_sync_chat_completion_is_costed_once(self) -> None:
        app = _SyncApp()
        tru_app = trulens_app.TruApp(
            app, app_name="sync_app", app_version="v1", main_method=app.query
        )
        with tru_app:
            self.assertEqual(app.query("q"), "Kojikun")

        self._assert_costed_once()
