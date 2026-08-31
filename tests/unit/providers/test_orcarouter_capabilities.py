# pyright: reportMissingImports=false, reportMissingModuleSource=false
import types

import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    # Ensure each test starts with a clean capability cache
    from trulens.providers.openai import (
        OpenAI,  # type: ignore[import-not-found]
    )

    OpenAI.clear_model_capabilities_cache()
    yield
    # And leave no residue for other modules
    OpenAI.clear_model_capabilities_cache()


@pytest.mark.optional
def test_defaults_to_orcarouter_gateway(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")

    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    provider = OrcaRouter()

    assert provider.model_engine == "openai/gpt-4o-mini"
    assert str(provider.endpoint.client.client.base_url) == (
        "https://api.orcarouter.ai/v1/"
    )
    assert provider.endpoint.client.client.api_key == "sk-orca-test"


@pytest.mark.optional
def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)

    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
        OrcaRouter()


@pytest.mark.optional
def test_respects_env_var_base_url(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    monkeypatch.setenv("ORCAROUTER_BASE_URL", "https://my-gateway.example/v1")

    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    provider = OrcaRouter(model_engine="anthropic/claude-sonnet-4.6")

    assert provider.model_engine == "anthropic/claude-sonnet-4.6"
    assert str(provider.endpoint.client.client.base_url) == (
        "https://my-gateway.example/v1/"
    )


@pytest.mark.optional
def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    monkeypatch.setenv("ORCAROUTER_BASE_URL", "https://env-gateway.example/v1")

    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    provider = OrcaRouter(base_url="https://explicit-gateway.example/v1")

    assert str(provider.endpoint.client.client.base_url) == (
        "https://explicit-gateway.example/v1/"
    )


@pytest.mark.optional
def test_reuses_openai_capability_logic(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")

    from trulens.providers.openai import (
        OpenAI,  # type: ignore[import-not-found]
    )
    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    provider = OrcaRouter(model_engine="openai/gpt-4o-mini")

    # OrcaRouter is API-compatible with OpenAI, so capability probing is
    # inherited unchanged. Whether the gateway actually honors structured
    # output is probed at runtime and cached per model id (see
    # `_call_with_capability_fallbacks`), so no static claim is made here.
    assert isinstance(provider, OpenAI)


class _DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[dict] = []

    class _Choices:
        def __init__(self, content: str):
            self.message = types.SimpleNamespace(content=content)

    class _Completion:
        def __init__(self, content: str):
            self.choices = [_DummyChatCompletions._Choices(content=content)]

    def create(self, *, messages, **kwargs):  # noqa: ANN001
        self.create_calls.append(dict(kwargs))
        # Return a 3/3 rating so `relevance` normalizes to 1.0.
        return _DummyChatCompletions._Completion(
            content='{"score": 3, "reason": "matches"}'
        )


class _DummyResponses:
    def parse(self, *, input, text_format, **kwargs):  # noqa: ANN001
        # Structured outputs are not probed on the gateway for this test; the
        # smoke test below exercises the chat.completions path.
        raise Exception("structured outputs unsupported")


class _DummyChat:
    def __init__(self, completions: _DummyChatCompletions):
        self.completions = completions


class _DummyClient:
    def __init__(
        self, responses: _DummyResponses, completions: _DummyChatCompletions
    ):
        self.responses = responses
        self.chat = _DummyChat(completions)


@pytest.mark.optional
def test_relevance_smoke_test(monkeypatch):
    """End-to-end feedback evaluation through the mocked gateway.

    Exercises `relevance` → `generate_score` → `_create_chat_completion`,
    confirming the default model id is sent and the score is parsed and
    normalized.
    """
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")

    from trulens.providers.orcarouter import (
        OrcaRouter,  # type: ignore[import-not-found]
    )

    provider = OrcaRouter()

    completions = _DummyChatCompletions()
    provider.endpoint.client = _DummyClient(_DummyResponses(), completions)

    score = provider.relevance("what is a cow?", "A cow is an animal.")

    assert score == pytest.approx(1.0)
    # The default model id must be forwarded to the gateway.
    assert completions.create_calls
    assert completions.create_calls[0]["model"] == "openai/gpt-4o-mini"
