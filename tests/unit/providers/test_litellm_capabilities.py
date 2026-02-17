# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os
from typing import Any, Dict

import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    try:
        from trulens.providers.litellm import (
            LiteLLM,  # type: ignore[import-not-found]
        )
    except Exception:
        yield
        return

    LiteLLM.clear_model_capabilities_cache()
    yield
    LiteLLM.clear_model_capabilities_cache()


class _DummyLiteLLM:
    """Monkeypatch target for litellm.completion to simulate unsupported params."""

    def __init__(self):
        self.calls: list[Dict[str, Any]] = []
        self.fail_on: Dict[str, bool] = {}

    class _Choices:
        def __init__(self, content: str):
            class _Msg:
                def __init__(self, content: str):
                    self.content = content

            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_DummyLiteLLM._Choices(content=content)]

    def completion(self, **kwargs):
        # record
        self.calls.append(dict(kwargs))
        for param, should_fail in self.fail_on.items():
            if should_fail and param in kwargs:
                raise TypeError(f"{param} is not allowed")
        return _DummyLiteLLM._Resp(content="ok")


def _make_provider(monkeypatch, *, model_engine: str = "gpt-4o-mini"):
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    # Patch litellm.completion used by provider
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod, "completion", lambda **kw: dummy.completion(**kw)
    )
    # get_supported_openai_params returns None/[] in many cases; keep default
    return LiteLLM(model_engine=model_engine), dummy


@pytest.mark.optional
def test_temperature_fallback_and_cached(monkeypatch):
    provider, dummy = _make_provider(monkeypatch)
    dummy.fail_on = {"temperature": True}

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], temperature=0.7
    )
    assert out == "ok"
    # first attempt with temperature, then without
    assert len(dummy.calls) == 2
    assert "temperature" in dummy.calls[0]
    assert "temperature" not in dummy.calls[1]

    dummy.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], temperature=0.9
    )
    assert out2 == "ok"
    assert len(dummy.calls) == 1
    assert "temperature" not in dummy.calls[0]


@pytest.mark.optional
def test_temperature_success_and_cached(monkeypatch):
    provider, dummy = _make_provider(monkeypatch)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], temperature=0.7
    )
    assert out == "ok"
    assert len(dummy.calls) == 1
    assert dummy.calls[0].get("temperature") == 0.7

    dummy.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], temperature=0.9
    )
    assert out2 == "ok"
    assert len(dummy.calls) == 1
    assert dummy.calls[0].get("temperature") == 0.9


@pytest.mark.optional
def test_reasoning_effort_fallback_and_cached(monkeypatch):
    provider, dummy = _make_provider(monkeypatch, model_engine="o1-mini")
    dummy.fail_on = {"reasoning_effort": True}

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], reasoning_effort="medium"
    )
    assert out == "ok"

    dummy.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], reasoning_effort="high"
    )
    assert out2 == "ok"
    assert len(dummy.calls) == 1
    assert "reasoning_effort" not in dummy.calls[0]


@pytest.mark.optional
def test_reasoning_effort_success_and_cached(monkeypatch):
    provider, dummy = _make_provider(monkeypatch, model_engine="o1-mini")

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], reasoning_effort="high"
    )
    assert out == "ok"
    assert len(dummy.calls) == 1
    assert dummy.calls[0].get("reasoning_effort") == "high"

    dummy.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], reasoning_effort="low"
    )
    assert out2 == "ok"
    assert len(dummy.calls) == 1
    assert dummy.calls[0].get("reasoning_effort") == "low"


@pytest.mark.optional
def test_is_reasoning_model_with_prefixed_name(monkeypatch):
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    # Prefixed provider ids should be parsed after '/'
    assert (
        LiteLLM(model_engine="snowflake/o3-mini")._is_reasoning_model() is True
    )
    assert (
        LiteLLM(
            model_engine="anthropic/claude-3-7-sonnet-thinking"
        )._is_reasoning_model()
        is True
    )
    assert (
        LiteLLM(model_engine="openai/gpt-4o-mini")._is_reasoning_model()
        is False
    )


# --- Tests for litellm routing params (api_base, api_key, etc.) ---


@pytest.mark.optional
def test_api_base_forwarded_via_direct_kwarg(monkeypatch):
    """api_base passed as a direct kwarg should be forwarded to
    litellm.completion() calls."""
    provider, dummy = _make_provider(monkeypatch)

    # Override completion_args after construction to inject api_base
    # the way the constructor should have stored it.
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    provider = LiteLLM(
        model_engine="gpt-4o-mini",
        api_base="http://custom-host:8080",
    )
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    assert dummy.calls[-1].get("api_base") == "http://custom-host:8080"


@pytest.mark.optional
def test_api_base_forwarded_via_completion_kwargs(monkeypatch):
    """api_base passed via completion_kwargs should be forwarded to
    litellm.completion() calls."""
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    provider = LiteLLM(
        model_engine="gpt-4o-mini",
        completion_kwargs={"api_base": "http://other-host:9090"},
    )
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    assert dummy.calls[-1].get("api_base") == "http://other-host:9090"


@pytest.mark.optional
def test_api_base_preserved_with_prefixed_model(monkeypatch):
    """api_base must survive the param-filtering step for models
    with a provider prefix like ollama/... (GH-1804)."""
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    provider = LiteLLM(
        model_engine="ollama/qwen2.5:72b",
        api_base="http://17.1.44.10:8952",
    )
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    assert dummy.calls[-1].get("api_base") == "http://17.1.44.10:8952"


@pytest.mark.optional
def test_completion_kwargs_takes_precedence_over_direct_kwarg(
    monkeypatch,
):
    """If the same routing param is in both completion_kwargs and
    **kwargs, completion_kwargs should win."""
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    provider = LiteLLM(
        model_engine="gpt-4o-mini",
        completion_kwargs={"api_base": "http://from-kwargs:1111"},
        api_base="http://from-direct:2222",
    )
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    # completion_kwargs value should take precedence
    assert dummy.calls[-1].get("api_base") == "http://from-kwargs:1111"


@pytest.mark.optional
def test_env_var_api_base_not_interfered_with(monkeypatch):
    """When api_base is set via env var (e.g. OLLAMA_API_BASE), TruLens
    must not interfere.  litellm reads the env var internally, so our
    code should simply not inject an api_base kwarg."""
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    # Set the env var that litellm checks for Ollama
    monkeypatch.setenv("OLLAMA_API_BASE", "http://remote-ollama:11434")

    provider = LiteLLM(model_engine="ollama/llama3.1:8b")
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    # No api_base kwarg should be injected by TruLens — litellm will
    # read OLLAMA_API_BASE internally from the environment.
    assert "api_base" not in dummy.calls[-1]
    # Verify the env var is actually set (litellm would read it)
    assert os.environ.get("OLLAMA_API_BASE") == "http://remote-ollama:11434"


@pytest.mark.optional
def test_explicit_api_base_overrides_env_var(monkeypatch):
    """When api_base is passed explicitly AND env var is set, the
    explicit param should be forwarded (litellm gives precedence to
    the param over the env var)."""
    from trulens.providers.litellm import (
        LiteLLM,  # type: ignore[import-not-found]
    )

    dummy = _DummyLiteLLM()
    import trulens.providers.litellm.provider as provider_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        provider_mod,
        "completion",
        lambda **kw: dummy.completion(**kw),
    )

    monkeypatch.setenv("OLLAMA_API_BASE", "http://env-host:11434")

    provider = LiteLLM(
        model_engine="ollama/llama3.1:8b",
        api_base="http://explicit-host:8080",
    )
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(dummy.calls) >= 1
    # Explicit param should be forwarded, taking priority over env var
    assert dummy.calls[-1].get("api_base") == "http://explicit-host:8080"
