# pyright: reportMissingImports=false, reportMissingModuleSource=false
from typing import Any, Dict, Optional

import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    # Ensure each test starts with a clean capability cache
    try:
        from trulens.providers.langchain import (
            Langchain,  # type: ignore[import-not-found]
        )
    except Exception:
        # If langchain provider isn't available, skip the reset
        yield
        return

    Langchain.clear_model_capabilities_cache()
    yield
    # And leave no residue for other modules
    Langchain.clear_model_capabilities_cache()


def _has_langchain_core() -> bool:
    try:
        import langchain_core  # noqa: F401

        return True
    except Exception:
        return False


class _DummyChatModel:
    """A minimal BaseChatModel stub recording kwargs in _generate and
    simulating unsupported params via TypeError.
    """

    def __init__(self, *, fail_on_params: Optional[Dict[str, bool]] = None):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration
        from langchain_core.outputs import ChatResult
        from pydantic import PrivateAttr

        class _Impl(BaseChatModel):  # type: ignore[misc]
            _fail_on_params: Dict[str, bool] = PrivateAttr(default_factory=dict)
            _calls: list[Dict[str, Any]] = PrivateAttr(default_factory=list)

            def __init__(
                self, fail_on_params: Optional[Dict[str, bool]] = None
            ):
                super().__init__()
                self._fail_on_params = fail_on_params or {}
                self._calls = []

            @property
            def _llm_type(self) -> str:  # noqa: D401
                return "dummy"

            def _generate(
                self, messages, stop=None, run_manager=None, **kwargs
            ):  # noqa: ANN001
                # Record seen kwargs for assertions
                self._calls.append(dict(kwargs))
                for param, should_fail in self._fail_on_params.items():
                    if should_fail and param in kwargs:
                        raise TypeError(f"{param} is not allowed")

                gen = ChatGeneration(message=AIMessage(content="ok"))
                return ChatResult(generations=[gen])

        self.impl = _Impl(fail_on_params=fail_on_params)

    @property
    def instance(self):
        return self.impl

    @property
    def calls(self):
        return self.impl._calls


def _make_provider(
    monkeypatch,
    *,
    model_engine: str = "",
    fail_on: Optional[Dict[str, bool]] = None,
):
    if not _has_langchain_core():
        pytest.skip(
            "langchain_core not installed; skipping optional LangChain provider tests."
        )

    from trulens.providers.langchain import (
        Langchain,  # type: ignore[import-not-found]
    )

    dummy_chat = _DummyChatModel(fail_on_params=fail_on or {})
    provider = Langchain(chain=dummy_chat.instance, model_engine=model_engine)
    return provider, dummy_chat


@pytest.mark.optional
def test_temperature_fallback_and_cached(monkeypatch):
    provider, dummy_chat = _make_provider(
        monkeypatch, fail_on={"temperature": True}
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], temperature=0.7
    )
    # Should succeed after retry without temperature
    assert out == "ok"
    calls = dummy_chat.calls
    # Two attempts: first with temperature, second without
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]

    # Reset call log
    dummy_chat.calls.clear()

    # Call again with temperature; cache should strip it before calling
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}], temperature=0.9
    )
    assert out2 == "ok"
    calls2 = dummy_chat.calls
    assert len(calls2) == 1
    assert "temperature" not in calls2[0]


@pytest.mark.optional
def test_temperature_success_and_cached(monkeypatch):
    provider, dummy_chat = _make_provider(monkeypatch)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], temperature=0.7
    )
    assert out == "ok"
    calls = dummy_chat.calls
    assert len(calls) == 1
    assert calls[0].get("temperature") == 0.7

    # Second call should also include temperature, using cached allows=True
    dummy_chat.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}], temperature=0.9
    )
    assert out2 == "ok"
    calls2 = dummy_chat.calls
    assert len(calls2) == 1
    assert calls2[0].get("temperature") == 0.9


@pytest.mark.optional
def test_reasoning_effort_fallback_and_cached(monkeypatch):
    # Use a reasoning model id so upstream logic keeps reasoning_effort
    provider, dummy_chat = _make_provider(
        monkeypatch, model_engine="o1-mini", fail_on={"reasoning_effort": True}
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], reasoning_effort="medium"
    )
    assert out == "ok"

    # Next call should not include reasoning_effort anymore
    dummy_chat.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], reasoning_effort="high"
    )
    assert out2 == "ok"
    calls = dummy_chat.calls
    assert len(calls) == 1
    assert "reasoning_effort" not in calls[0]


@pytest.mark.optional
def test_reasoning_effort_success_and_cached(monkeypatch):
    # Use a reasoning model so provider sets/keeps reasoning_effort
    provider, dummy_chat = _make_provider(monkeypatch, model_engine="o1-mini")

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], reasoning_effort="high"
    )
    assert out == "ok"
    calls = dummy_chat.calls
    assert len(calls) == 1
    assert calls[0].get("reasoning_effort") == "high"

    # Next call should also include reasoning_effort using cached supports=True
    dummy_chat.calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}], reasoning_effort="low"
    )
    assert out2 == "ok"
    calls2 = dummy_chat.calls
    assert len(calls2) == 1
    assert calls2[0].get("reasoning_effort") == "low"


@pytest.mark.optional
def test_is_reasoning_model_detection():
    if not _has_langchain_core():
        pytest.skip(
            "langchain_core not installed; skipping optional LangChain provider tests."
        )

    from trulens.providers.langchain import (
        Langchain,  # type: ignore[import-not-found]
    )

    dummy_chat = _DummyChatModel()

    # Model names indicating reasoning
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="gpt-5-mini"
        )._is_reasoning_model()
        is True
    )
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="o1-mini"
        )._is_reasoning_model()
        is True
    )
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="deepseek-r1"
        )._is_reasoning_model()
        is True
    )
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="my-reasoning-model"
        )._is_reasoning_model()
        is True
    )
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="my-thinking-model"
        )._is_reasoning_model()
        is True
    )
    # Non-reasoning
    assert (
        Langchain(
            chain=dummy_chat.instance, model_engine="gpt-4o-mini"
        )._is_reasoning_model()
        is False
    )
