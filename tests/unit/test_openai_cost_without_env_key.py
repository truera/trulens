"""OTEL cost tracking for OpenAI must not need OPENAI_API_KEY in the environment.

An app can pass its key to the client directly, as in
`openai.OpenAI(api_key=...)`. Cost tracking only reads the usage on the
response, so it has to work without the environment variable too.
"""

import os
import unittest
from unittest import mock

import pytest

pytest.importorskip("openai")
pytest.importorskip("trulens.providers.openai")

import openai
from openai.types.completion_usage import CompletionUsage
from trulens.core.utils.python import SingletonPerNameMeta
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai.endpoint import OpenAICostComputer
from trulens.providers.openai.endpoint import OpenAIEndpoint

_MODEL = "gpt-3.5-turbo"
"""A model langchain knows the price of, so the cost is not zero."""


def _chat_completion() -> openai.types.chat.ChatCompletion:
    return openai.types.chat.ChatCompletion(
        id="chatcmpl-test",
        created=0,
        model=_MODEL,
        object="chat.completion",
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=1200, completion_tokens=300, total_tokens=1500
        ),
    )


class TestOpenAICostWithoutEnvKey(unittest.TestCase):
    def setUp(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        self._env = mock.patch.dict(os.environ, env, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)

        # A cached endpoint from another test would hide the bug, since it
        # already has a client.
        instances = SingletonPerNameMeta._singleton_instances
        key = (f"{OpenAIEndpoint.__module__}.{OpenAIEndpoint.__name__}", None)
        saved = instances.pop(key, None)
        if saved is not None:
            self.addCleanup(instances.__setitem__, key, saved)

    def test_handle_response_without_env_key(self) -> None:
        with self.assertRaises(openai.OpenAIError):
            openai.OpenAI()

        attrs = OpenAICostComputer.handle_response(_chat_completion())

        self.assertEqual(attrs[SpanAttributes.COST.NUM_TOKENS], 1500)
        self.assertEqual(attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS], 1200)
        self.assertEqual(attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 300)
        self.assertEqual(attrs[SpanAttributes.COST.MODEL], _MODEL)
        self.assertGreater(attrs[SpanAttributes.COST.COST], 0.0)

    def test_handle_response_does_not_create_an_endpoint(self) -> None:
        OpenAICostComputer.handle_response(_chat_completion())

        key = (f"{OpenAIEndpoint.__module__}.{OpenAIEndpoint.__name__}", None)
        self.assertNotIn(key, SingletonPerNameMeta._singleton_instances)


if __name__ == "__main__":
    unittest.main()
