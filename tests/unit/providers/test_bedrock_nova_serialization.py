# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Regression tests for Bedrock Amazon Nova request/response serialization.

The default Bedrock model is ``amazon.nova-lite-v1:0`` (set in #2511). Nova uses
the Messages API schema for ``InvokeModel`` (``schemaVersion: messages-v1``,
``messages``/``inferenceConfig``) and returns text at
``output.message.content[0].text`` -- not the legacy Titan
``inputText``/``results[0].outputText`` schema used by other ``amazon.*`` models.

References:
- https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
- https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html

These tests mock the boto3 ``invoke_model`` seam, so no AWS account or
credentials are required.
"""

import json
from typing import Any, Dict, List

import pytest


class _DummyBody:
    """Mimics botocore's StreamingBody: a single ``.read()`` of JSON bytes."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _DummyBedrockClient:
    """Stands in for the boto3 ``bedrock-runtime`` client.

    Records each ``invoke_model`` call and returns a canned response body, so no
    AWS call or credentials are needed.
    """

    def __init__(self, response_payload: Dict[str, Any]):
        self._payload = json.dumps(response_payload).encode()
        self.calls: List[Dict[str, Any]] = []

    def invoke_model(self, *, body, modelId, accept, contentType):  # noqa: N803
        self.calls.append({
            "body": body,
            "modelId": modelId,
            "accept": accept,
            "contentType": contentType,
        })
        return {"body": _DummyBody(self._payload)}


# A canonical Nova ``InvokeModel`` response (Messages API shape).
_NOVA_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "nova-ok"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
}


def _make_provider(model_id: str, response_payload: Dict[str, Any]):
    from trulens.providers.bedrock import (
        Bedrock,  # type: ignore[import-not-found]
    )

    provider = Bedrock(model_id=model_id)
    # Replace the boto3-backed client with our dummy (no AWS, no credentials).
    provider.endpoint.client = _DummyBedrockClient(response_payload)
    return provider


@pytest.mark.optional
def test_default_model_is_nova():
    """The default Bedrock model is Nova Lite (#2511)."""
    from trulens.providers.bedrock import (
        Bedrock,  # type: ignore[import-not-found]
    )

    assert Bedrock.DEFAULT_MODEL_ID == "amazon.nova-lite-v1:0"


@pytest.mark.optional
@pytest.mark.parametrize(
    "model_id",
    [
        "amazon.nova-lite-v1:0",  # default
        "amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",  # cross-region inference profile id
    ],
)
def test_nova_request_uses_messages_api_schema(model_id):
    """Nova serializes to the Messages API schema, not the Titan schema."""
    provider = _make_provider(model_id, _NOVA_RESPONSE)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert out == "nova-ok"

    calls = provider.endpoint.client.calls
    assert len(calls) == 1
    assert calls[0]["modelId"] == model_id

    body = json.loads(calls[0]["body"])
    # Messages API schema (Nova) -- must NOT be the Titan text schema.
    assert body["schemaVersion"] == "messages-v1"
    assert "messages" in body
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"][0]["text"]  # non-empty prompt text
    assert "inferenceConfig" in body
    assert body["inferenceConfig"]["maxTokens"] == 4095
    # The legacy Titan keys must be absent for Nova.
    assert "inputText" not in body
    assert "textGenerationConfig" not in body


@pytest.mark.optional
def test_nova_response_parses_messages_api_shape():
    """Nova response text is read from output.message.content[0].text."""
    provider = _make_provider("amazon.nova-lite-v1:0", _NOVA_RESPONSE)

    out = provider._create_chat_completion(prompt="hello")

    assert out == "nova-ok"


@pytest.mark.optional
def test_titan_still_uses_legacy_schema():
    """Non-Nova amazon.* models keep the Titan text-generation schema."""
    provider = _make_provider(
        "amazon.titan-text-express-v1",
        {"results": [{"outputText": "titan-ok"}]},
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert out == "titan-ok"

    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert "inputText" in body
    assert "textGenerationConfig" in body
    assert "schemaVersion" not in body
