# pyright: reportMissingImports=false, reportMissingModuleSource=false
import types
from typing import Any, Dict

import pydantic
import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    # Ensure each test starts with a clean capability cache
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    Ollama.clear_model_capabilities_cache()
    yield
    Ollama.clear_model_capabilities_cache()


class _DummyResponses:
    """Simulates Ollama's OpenAI-compatible server, which does not expose
    the `/v1/responses` endpoint."""

    def __init__(self):
        self.parse_calls = 0

    def parse(self, *, input, text_format, **kwargs):  # noqa: ANN001
        self.parse_calls += 1
        raise Exception("404 page not found: responses api not supported")


class _DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[Dict[str, Any]] = []

    class _Choices:
        def __init__(self, content: str):
            self.message = types.SimpleNamespace(content=content)

    class _Completion:
        def __init__(self, content: str):
            self.choices = [_DummyChatCompletions._Choices(content=content)]

    def create(self, *, messages, **kwargs):  # noqa: ANN001
        self.create_calls.append(dict(kwargs))
        return _DummyChatCompletions._Completion(content="ok")


class _DummyChat:
    def __init__(self, completions: _DummyChatCompletions):
        self.completions = completions


class _DummyClient:
    def __init__(self):
        self.responses = _DummyResponses()
        self.chat = _DummyChat(_DummyChatCompletions())


class _ParsedModel(pydantic.BaseModel):
    value: str


@pytest.mark.optional
def test_defaults_to_local_ollama_server(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama()

    assert provider.model_engine == "llama3.2"
    assert str(provider.endpoint.client.client.base_url) == (
        "http://localhost:11434/v1/"
    )
    assert provider.endpoint.client.client.api_key == "ollama"


@pytest.mark.optional
def test_respects_env_var_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama-host:11434/v1")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="qwen2.5")

    assert provider.model_engine == "qwen2.5"
    assert str(provider.endpoint.client.client.base_url) == (
        "http://my-ollama-host:11434/v1/"
    )


@pytest.mark.optional
def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:11434/v1")

    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(base_url="http://explicit-host:11434/v1")

    assert str(provider.endpoint.client.client.base_url) == (
        "http://explicit-host:11434/v1/"
    )


@pytest.mark.optional
def test_falls_back_to_chat_completions_when_responses_api_missing(
    monkeypatch,
):
    # Ollama's OpenAI-compatible server does not implement the Responses
    # API, so structured-output/CFG probing must gracefully fall back to
    # the Chat Completions endpoint rather than raising.
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")
    dummy_client = _DummyClient()
    provider.endpoint.client = dummy_client

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_ParsedModel,
    )

    # responses.parse() was tried once, found unavailable, then the call
    # fell back (and stayed cached) to chat.completions.create().
    assert out == "ok"
    assert dummy_client.responses.parse_calls == 1
    assert len(dummy_client.chat.completions.create_calls) == 1

    # A second call should skip the unavailable Responses API entirely.
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}],
        response_format=_ParsedModel,
    )
    assert out2 == "ok"
    assert dummy_client.responses.parse_calls == 1
    assert len(dummy_client.chat.completions.create_calls) == 2
