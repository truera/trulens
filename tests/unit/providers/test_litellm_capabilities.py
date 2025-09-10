# pyright: reportMissingImports=false, reportMissingModuleSource=false
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
