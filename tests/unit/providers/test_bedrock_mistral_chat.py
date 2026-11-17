# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Bedrock Mistral request/response schema (#3081).

Recent Mistral models (Large 3, Ministral 3, Magistral) take the chat schema
(``messages`` in, ``choices[0].message.content`` out); older ones keep the
text-completion schema (``prompt`` in, ``outputs[0].text`` out).

The boto3 ``invoke_model`` seam is mocked, so no AWS account is needed.
"""

import json
from typing import Any, Dict, List

import pytest


class _DummyBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _DummyBedrockClient:
    def __init__(self, response_payload: Dict[str, Any]):
        self._payload = json.dumps(response_payload).encode()
        self.calls: List[Dict[str, Any]] = []

    def invoke_model(self, *, body, modelId, accept, contentType):  # noqa: N803
        self.calls.append({"body": body, "modelId": modelId})
        return {"body": _DummyBody(self._payload)}


def _make_provider(model_id: str, response_payload: Dict[str, Any]):
    from trulens.providers.bedrock import Bedrock

    provider = Bedrock(model_id=model_id)
    provider.endpoint.client = _DummyBedrockClient(response_payload)
    return provider


@pytest.mark.optional
@pytest.mark.parametrize(
    "model_id",
    [
        "mistral.mistral-large-3-675b-instruct",
        "mistral.ministral-3-8b-instruct",
        "us.mistral.magistral-small-2509",
    ],
)
def test_new_mistral_uses_chat_schema(model_id):
    provider = _make_provider(
        model_id,
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert out == "ok"
    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "prompt" not in body


@pytest.mark.optional
@pytest.mark.parametrize(
    "model_id",
    [
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0",
    ],
)
def test_legacy_mistral_keeps_prompt_schema(model_id):
    provider = _make_provider(model_id, {"outputs": [{"text": "legacy-ok"}]})

    out = provider._create_chat_completion(prompt="hello")

    assert out == "legacy-ok"
    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert "prompt" in body
    assert "messages" not in body
