"""Regression tests: the Anthropic provider must work with anthropic>=1.0.

anthropic 1.0.0 removed ``temperature``, ``top_p`` and ``top_k`` from the
signature of ``Messages.create`` (see the SDK's MIGRATION.md; models that still
take them get them through ``extra_body``). The provider used to pass
``temperature=0.0`` to ``messages.create`` for every non-reasoning model,
including the default ``claude-sonnet-4-6``, so every feedback call failed with
a ``TypeError`` that ``run_in_pace`` retried and then re-raised.

These tests drive a real ``anthropic.Anthropic`` client against a mock HTTP
transport, so the SDK's own method signature is exercised and the request body
that would reach the API is checked.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.optional

anthropic = pytest.importorskip("anthropic")

# anthropic>=1.0 is built on httpx2, earlier versions on httpx.
if int(anthropic.__version__.split(".")[0]) >= 1:
    httpx = pytest.importorskip("httpx2")
else:
    httpx = pytest.importorskip("httpx")


def _message_json(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


@pytest.fixture
def requests() -> list[dict]:
    return []


@pytest.fixture
def provider(requests):
    from trulens.providers.anthropic import Anthropic

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_message_json("2"))

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return Anthropic(client=client, model_engine="claude-sonnet-4-6")


def test_chat_completion_with_default_temperature(provider, requests):
    assert provider._create_chat_completion(prompt="Rate this.") == "2"
    assert requests[0]["temperature"] == 0.0


def test_chat_completion_with_explicit_temperature(provider, requests):
    assert (
        provider._create_chat_completion(prompt="Rate this.", temperature=0.7)
        == "2"
    )
    assert requests[0]["temperature"] == 0.7


def test_chat_completion_merges_user_extra_body(provider, requests):
    assert (
        provider._create_chat_completion(
            prompt="Rate this.", top_k=5, extra_body={"metadata_tag": "x"}
        )
        == "2"
    )
    body = requests[0]
    assert body["temperature"] == 0.0
    assert body["top_k"] == 5
    assert body["metadata_tag"] == "x"
