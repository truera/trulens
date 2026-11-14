"""Exercise credential routing and evaluation without third-party requests."""

import json

import httpx
import pytest
from trulens.providers.api_route import provider as api_route_provider
from trulens.providers.openai import provider as openai_provider


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    """Prevent ambient keys and capability caches from affecting tests."""
    monkeypatch.delenv("API_ROUTE_API_KEY", raising=False)
    monkeypatch.delenv("API_ROUTE_BASE_URL", raising=False)
    openai_provider.OpenAI.clear_model_capabilities_cache()
    yield
    openai_provider.OpenAI.clear_model_capabilities_cache()


def test_default_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("API_ROUTE_API_KEY", "route-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    provider = api_route_provider.APIRoute()
    assert provider.model_engine == "gpt-6.1-sol"
    assert str(provider.endpoint.client.client.base_url) == (
        "https://global.api-route.com/v1/"
    )
    assert provider.endpoint.client.client.api_key == "route-key"


def test_explicit_settings_override_environment(monkeypatch):
    monkeypatch.setenv("API_ROUTE_API_KEY", "environment-key")
    monkeypatch.setenv("API_ROUTE_BASE_URL", "https://environment.example/v1")
    provider = api_route_provider.APIRoute(
        api_key="explicit-key",
        base_url="https://explicit.example/v1",
        model_engine="model-from-key",
    )
    assert provider.endpoint.client.client.api_key == "explicit-key"
    assert str(provider.endpoint.client.client.base_url) == (
        "https://explicit.example/v1/"
    )
    assert provider.model_engine == "model-from-key"


def test_environment_endpoint(monkeypatch):
    monkeypatch.setenv("API_ROUTE_BASE_URL", "https://environment.example/v1")
    provider = api_route_provider.APIRoute(api_key="route-key")
    assert str(provider.endpoint.client.client.base_url) == (
        "https://environment.example/v1/"
    )


@pytest.mark.parametrize("api_key", [None, ""])
def test_missing_route_key_never_falls_back_to_openai(monkeypatch, api_key):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    with pytest.raises(ValueError, match="API_ROUTE_API_KEY"):
        api_route_provider.APIRoute(api_key=api_key)


def test_evaluation_routes_requests_and_normalizes_score():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                404,
                json={"error": {"message": "Responses API unavailable"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"score": 3}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider = api_route_provider.APIRoute(
            api_key="route-key", http_client=client
        )
        assert provider.relevance(
            "What is a cow?", "A cow is an animal."
        ) == pytest.approx(1.0)
    assert requests
    assert all(
        request.url.host == "global.api-route.com" for request in requests
    )
    assert all(
        request.headers["authorization"] == "Bearer route-key"
        for request in requests
    )
    chat_requests = [
        request
        for request in requests
        if request.url.path.endswith("/chat/completions")
    ]
    assert chat_requests
    assert json.loads(chat_requests[-1].content)["model"] == "gpt-6.1-sol"
    assert not provider.reports_costs


def test_moderation_fails_without_network():
    def unexpected_request(request):
        pytest.fail("Moderation must not make an HTTP request")

    with httpx.Client(
        transport=httpx.MockTransport(unexpected_request)
    ) as client:
        provider = api_route_provider.APIRoute(
            api_key="route-key", http_client=client
        )
        with pytest.raises(NotImplementedError, match="Moderation"):
            provider.moderation_hate("text")
