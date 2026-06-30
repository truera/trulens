# pyright: reportMissingImports=false, reportMissingModuleSource=false
from typing import Any, Dict, List

import pytest


def _stub_session():
    """A no-op Snowflake ``Session`` instance.

    ``Cortex`` is a pydantic model that validates ``snowpark_session`` against
    the real ``Session`` type, so the stub must subclass it — but its
    ``__init__`` is bypassed so no real Snowflake connection is opened.
    """
    from snowflake.snowpark import Session  # type: ignore[import-not-found]

    class _NoOpSession(Session):
        def __init__(self):
            pass

    return _NoOpSession()


def _make_provider(monkeypatch, model_engine=None, complete_return="cortex-ok"):
    # Import here to avoid a heavy import at collection time.
    from trulens.providers.cortex import (
        Cortex,  # type: ignore[import-not-found]
    )
    from trulens.providers.cortex import (
        provider as cortex_provider,  # type: ignore[import-not-found]
    )

    calls: List[Dict[str, Any]] = []

    def fake_complete(*, model, prompt, options, session, stream, timeout):
        calls.append({
            "model": model,
            "prompt": prompt,
            "options": options,
            "session": session,
            "stream": stream,
            "timeout": timeout,
        })
        return complete_return

    # Replace the module-level `complete` the provider imported from snowflake,
    # so no Snowflake account or network call is needed.
    monkeypatch.setattr(cortex_provider, "complete", fake_complete)

    provider = Cortex(
        snowpark_session=_stub_session(), model_engine=model_engine
    )
    return provider, calls


@pytest.mark.optional
def test_default_model_engine():
    """No model_engine -> the documented llama3.3-70b default."""
    from trulens.providers.cortex import (
        Cortex,  # type: ignore[import-not-found]
    )

    assert Cortex.DEFAULT_MODEL_ENGINE == "llama3.3-70b"


@pytest.mark.optional
def test_messages_forwarded_to_complete(monkeypatch):
    """messages pass through to complete() with the default model and the
    session, and streaming is disabled."""
    provider, calls = _make_provider(monkeypatch)
    msgs = [{"role": "user", "content": "hi"}]
    out = provider._create_chat_completion(messages=msgs)

    assert out == "cortex-ok"
    assert len(calls) == 1
    assert calls[0]["model"] == "llama3.3-70b"
    assert calls[0]["prompt"] == msgs
    assert calls[0]["stream"] is False
    assert calls[0]["session"] is provider.snowpark_session


@pytest.mark.optional
def test_explicit_model_engine_used(monkeypatch):
    """An explicit model_engine overrides the default."""
    provider, calls = _make_provider(monkeypatch, model_engine="mistral-large2")
    provider._create_chat_completion(
        messages=[{"role": "user", "content": "x"}]
    )
    assert calls[0]["model"] == "mistral-large2"


@pytest.mark.optional
def test_prompt_is_wrapped_as_system_message(monkeypatch):
    """A bare prompt is wrapped into a single system message."""
    provider, calls = _make_provider(monkeypatch)
    provider._create_chat_completion(prompt="summarize")
    assert calls[0]["prompt"] == [{"role": "system", "content": "summarize"}]


@pytest.mark.optional
def test_requires_prompt_or_messages(monkeypatch):
    """Neither prompt nor messages supplied is a ValueError."""
    provider, _ = _make_provider(monkeypatch)
    with pytest.raises(ValueError):
        provider._create_chat_completion()


@pytest.mark.optional
def test_response_format_parsed(monkeypatch):
    """A response_format with valid JSON output is parsed into the model."""
    import pydantic

    class _Out(pydantic.BaseModel):
        value: str

    provider, _ = _make_provider(monkeypatch, complete_return='{"value": "ok"}')
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "x"}], response_format=_Out
    )
    assert isinstance(out, _Out)
    assert out.value == "ok"


@pytest.mark.optional
def test_response_format_parse_failure_returns_raw(monkeypatch):
    """If the output is not valid JSON for the schema, the raw string is
    returned rather than raising."""
    import pydantic

    class _Out(pydantic.BaseModel):
        value: str

    provider, _ = _make_provider(monkeypatch, complete_return="not json")
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "x"}], response_format=_Out
    )
    assert out == "not json"
