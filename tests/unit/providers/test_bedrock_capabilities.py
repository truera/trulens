# pyright: reportMissingImports=false, reportMissingModuleSource=false
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

    Records each ``invoke_model`` call and returns a canned, model-family-shaped
    response body, so no AWS call or credentials are needed.
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


def _make_provider(model_id: str, response_payload: Dict[str, Any]):
    # Import here to avoid a heavy import at collection time.
    from trulens.providers.bedrock import (
        Bedrock,  # type: ignore[import-not-found]
    )

    provider = Bedrock(model_id=model_id)
    # Replace the boto3-backed client with our dummy (no AWS, no credentials).
    provider.endpoint.client = _DummyBedrockClient(response_payload)
    return provider


# (model_id, response_payload, expected_text, request_body_key)
_FAMILY_CASES = [
    (
        "amazon.nova-lite-v1:0",
        {"results": [{"outputText": "amazon-ok"}]},
        "amazon-ok",
        "inputText",
    ),
    (
        "cohere.command-text-v14",
        {"generations": [{"text": "cohere-ok"}]},
        "cohere-ok",
        "prompt",
    ),
    (
        "ai21.j2-ultra-v1",
        {"completions": [{"data": {"text": "ai21-ok"}}]},
        "ai21-ok",
        "prompt",
    ),
    (
        "mistral.mistral-7b-instruct-v0:2",
        {"output": [{"text": "mistral-ok"}]},
        "mistral-ok",
        "prompt",
    ),
    (
        "meta.llama3-8b-instruct-v1:0",
        {"generation": "meta-ok"},
        "meta-ok",
        "prompt",
    ),
]


@pytest.mark.optional
@pytest.mark.parametrize("model_id,payload,expected,body_key", _FAMILY_CASES)
def test_model_family_request_and_response(
    model_id, payload, expected, body_key
):
    """Each model family routes to a family-specific request body and parses
    its own response shape back to the generated text."""
    provider = _make_provider(model_id, payload)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert out == expected

    calls = provider.endpoint.client.calls
    assert len(calls) == 1
    assert calls[0]["modelId"] == model_id
    assert calls[0]["accept"] == "application/json"
    assert calls[0]["contentType"] == "application/json"
    body = json.loads(calls[0]["body"])
    assert body_key in body


@pytest.mark.optional
def test_amazon_text_generation_config():
    """The amazon family nests generation params under textGenerationConfig."""
    provider = _make_provider(
        "amazon.titan-text-express-v1", {"results": [{"outputText": "ok"}]}
    )
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert out == "ok"
    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert "textGenerationConfig" in body
    assert body["textGenerationConfig"]["maxTokenCount"] == 4095


@pytest.mark.optional
def test_anthropic_request_shape():
    """Anthropic-on-Bedrock stamps the bedrock anthropic_version and splits the
    system message from the remaining messages."""
    provider = _make_provider(
        "anthropic.claude-3-sonnet-20240229-v1:0",
        {"content": [{"text": "claude-ok"}]},
    )
    out = provider._create_chat_completion(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out == "claude-ok"
    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.optional
def test_anthropic_requires_messages():
    """The anthropic path is messages-only; a bare prompt is rejected."""
    provider = _make_provider(
        "anthropic.claude-3-haiku-20240307-v1:0", {"content": [{"text": "ok"}]}
    )
    with pytest.raises(ValueError):
        provider._create_chat_completion(prompt="hi")


@pytest.mark.optional
def test_unsupported_model_family_raises():
    """An unrecognized model family is an explicit NotImplementedError."""
    provider = _make_provider(
        "unknown.model-v1", {"results": [{"outputText": "ok"}]}
    )
    with pytest.raises(NotImplementedError):
        provider._create_chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )


@pytest.mark.optional
def test_requires_messages_or_prompt():
    """Neither messages nor prompt supplied is a ValueError."""
    provider = _make_provider(
        "amazon.nova-lite-v1:0", {"results": [{"outputText": "ok"}]}
    )
    with pytest.raises(ValueError):
        provider._create_chat_completion()


@pytest.mark.optional
def test_messages_are_concatenated_into_input_text():
    """Multiple messages are flattened to a single 'role: content' string."""
    provider = _make_provider(
        "amazon.nova-lite-v1:0", {"results": [{"outputText": "ok"}]}
    )
    provider._create_chat_completion(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    body = json.loads(provider.endpoint.client.calls[0]["body"])
    assert body["inputText"] == "user: first assistant: second"


@pytest.mark.optional
def test_default_model_id_is_amazon_nova():
    """Constructed with no model_id, the provider defaults to amazon nova."""
    from trulens.providers.bedrock import (
        Bedrock,  # type: ignore[import-not-found]
    )

    provider = Bedrock()
    assert (
        provider.model_id == Bedrock.DEFAULT_MODEL_ID == "amazon.nova-lite-v1:0"
    )
