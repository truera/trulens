# pyright: reportMissingImports=false, reportMissingModuleSource=false
import types
from typing import Any, Dict, Optional

import pydantic
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


class _ParsedModel(pydantic.BaseModel):
    value: str


class _DummyResponses:
    def __init__(self, *, should_support: bool):
        self.should_support = should_support
        self.parse_calls = 0

    def parse(self, *, input, text_format, **kwargs):  # noqa: ANN001
        self.parse_calls += 1
        if self.should_support:
            # Return object that has `output_parsed` as the pydantic model instance
            class _R:
                def __init__(self, parsed):
                    self.output_parsed = parsed

            return _R(text_format(value="ok"))
        raise Exception("response_format unsupported")


class _DummyChatCompletions:
    def __init__(self, *, fail_on_params: Optional[Dict[str, bool]] = None):
        # e.g., {"temperature": True, "reasoning_effort": True}
        self.fail_on_params = fail_on_params or {}
        self.create_calls: list[Dict[str, Any]] = []  # list of kwargs seen

    class _Choices:
        def __init__(self, content: str):
            self.message = types.SimpleNamespace(content=content)

    class _Completion:
        def __init__(self, content: str):
            self.choices = [_DummyChatCompletions._Choices(content=content)]

    def create(self, *, messages, **kwargs):  # noqa: ANN001
        # record seen kwargs for assertions
        self.create_calls.append(dict(kwargs))

        for param, should_fail in self.fail_on_params.items():
            if should_fail and param in kwargs:
                raise Exception(f"{param} is not allowed")

        return _DummyChatCompletions._Completion(content="ok")


class _DummyChat:
    def __init__(self, completions: _DummyChatCompletions):
        self.completions = completions


class _DummyClient:
    def __init__(
        self, responses: _DummyResponses, completions: _DummyChatCompletions
    ):
        self.responses = responses
        self.chat = _DummyChat(completions)


def _make_provider(monkeypatch, model_engine: str = "gpt-4o-mini"):
    # Import here to avoid heavy imports at module import time
    from trulens.providers.openai import (
        OpenAI,  # type: ignore[import-not-found]
    )

    provider = OpenAI(model_engine=model_engine)

    # Replace endpoint.client with our dummy client
    dummy_responses = _DummyResponses(should_support=True)
    dummy_completions = _DummyChatCompletions()
    provider.endpoint.client = _DummyClient(dummy_responses, dummy_completions)
    return provider


@pytest.mark.optional
def test_structured_outputs_success_then_cached(monkeypatch):
    provider = _make_provider(monkeypatch)

    # Ensure responses.parse succeeds
    provider.endpoint.client.responses.should_support = True

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_ParsedModel,
    )
    assert isinstance(out, _ParsedModel)
    assert out.value == "ok"
    # parse called once
    assert provider.endpoint.client.responses.parse_calls == 1

    # Call again; should short-circuit to responses path again
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}],
        response_format=_ParsedModel,
    )
    assert isinstance(out2, _ParsedModel)
    assert provider.endpoint.client.responses.parse_calls == 2


@pytest.mark.optional
def test_structured_outputs_fallback_and_cached(monkeypatch):
    provider = _make_provider(monkeypatch)

    # First, make structured outputs unsupported
    provider.endpoint.client.responses.should_support = False

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_ParsedModel,
    )
    assert out == "ok"  # fell back to chat.completions
    # parse called once, then fallback used chat.create
    assert provider.endpoint.client.responses.parse_calls == 1
    assert len(provider.endpoint.client.chat.completions.create_calls) == 1

    # Reset counters, call again; cache should skip responses.parse
    provider.endpoint.client.responses.parse_calls = 0
    provider.endpoint.client.chat.completions.create_calls.clear()

    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}],
        response_format=_ParsedModel,
    )
    assert out2 == "ok"
    assert provider.endpoint.client.responses.parse_calls == 0  # cached skip
    assert len(provider.endpoint.client.chat.completions.create_calls) == 1


@pytest.mark.optional
def test_temperature_fallback_and_cached(monkeypatch):
    provider = _make_provider(monkeypatch)

    # Configure chat.create to fail when temperature is present
    provider.endpoint.client.chat.completions.fail_on_params = {
        "temperature": True
    }

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
    )
    # Should succeed after retry without temperature
    assert out == "ok"
    calls = provider.endpoint.client.chat.completions.create_calls
    assert len(calls) >= 1
    # First attempt had temperature
    assert any("temperature" in call for call in calls)

    # Reset call log
    provider.endpoint.client.chat.completions.create_calls.clear()

    # Call again with temperature; cache should strip it before calling
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}],
        temperature=0.9,
    )
    assert out2 == "ok"
    calls2 = provider.endpoint.client.chat.completions.create_calls
    assert len(calls2) == 1
    assert "temperature" not in calls2[0]


@pytest.mark.optional
def test_temperature_success_and_cached(monkeypatch):
    provider = _make_provider(monkeypatch)

    # Ensure temperature is allowed (default)
    provider.endpoint.client.chat.completions.fail_on_params = {}

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
    )
    assert out == "ok"
    calls = provider.endpoint.client.chat.completions.create_calls
    assert len(calls) == 1
    assert "temperature" in calls[0]
    assert calls[0]["temperature"] == 0.7

    # Second call should also include temperature, using cached allows=True
    provider.endpoint.client.chat.completions.create_calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi again"}],
        temperature=0.9,
    )
    assert out2 == "ok"
    calls2 = provider.endpoint.client.chat.completions.create_calls
    assert len(calls2) == 1
    assert calls2[0]["temperature"] == 0.9


@pytest.mark.optional
def test_reasoning_effort_fallback_and_cached(monkeypatch):
    # Use a reasoning model id so upstream logic keeps reasoning_effort
    provider = _make_provider(monkeypatch, model_engine="o1-mini")

    provider.endpoint.client.chat.completions.fail_on_params = {
        "reasoning_effort": True
    }

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="medium",
    )
    assert out == "ok"

    # Next call should not include reasoning_effort anymore
    provider.endpoint.client.chat.completions.create_calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}],
        reasoning_effort="high",
    )
    assert out2 == "ok"
    calls = provider.endpoint.client.chat.completions.create_calls
    assert len(calls) == 1
    assert "reasoning_effort" not in calls[0]


@pytest.mark.optional
def test_reasoning_effort_success_and_cached(monkeypatch):
    # Use a reasoning model so provider sets/keeps reasoning_effort
    provider = _make_provider(monkeypatch, model_engine="o1-mini")

    # Ensure reasoning_effort is allowed
    provider.endpoint.client.chat.completions.fail_on_params = {}

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )
    assert out == "ok"
    calls = provider.endpoint.client.chat.completions.create_calls
    assert len(calls) == 1
    assert calls[0].get("reasoning_effort") == "high"

    # Next call should also include reasoning_effort using cached supports=True
    provider.endpoint.client.chat.completions.create_calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}],
        reasoning_effort="low",
    )
    assert out2 == "ok"
    calls2 = provider.endpoint.client.chat.completions.create_calls
    assert len(calls2) == 1
    assert calls2[0].get("reasoning_effort") == "low"


@pytest.mark.optional
def test_is_reasoning_model_gpt5():
    from trulens.providers.openai import OpenAI

    assert OpenAI(model_engine="gpt-5-mini")._is_reasoning_model() is True
    assert OpenAI(model_engine="gpt-5")._is_reasoning_model() is True
    assert OpenAI(model_engine="gpt-4o-mini")._is_reasoning_model() is False


class _DummyResponsesWithCreate:
    def __init__(self, *, should_succeed: bool, tool_input: str = "ok_cfg"):
        self.should_succeed = should_succeed
        self.tool_input = tool_input
        self.create_calls = 0
        self.parse_calls = 0

    # Structured outputs path should be bypassed or set unsupported
    def parse(self, *args, **kwargs):  # noqa: ANN001
        self.parse_calls += 1
        raise Exception("structured outputs unsupported")

    def create(self, *args, **kwargs):  # noqa: ANN001
        self.create_calls += 1
        if not self.should_succeed:
            raise Exception("cfg unsupported")

        class _Item:
            def __init__(self, text: str):
                self.type = "tool"
                self.input = text

        class _Response:
            def __init__(self, items):
                self.output = items

        return _Response([_Item(self.tool_input)])


@pytest.mark.optional
def test_cfg_success_then_cached(monkeypatch):
    # Use a gpt-5 model so CFG auto-enable is considered
    provider = _make_provider(monkeypatch, model_engine="gpt-5-mini")

    # Swap in a responses client that supports create() with grammar tool
    provider.endpoint.client.responses = _DummyResponsesWithCreate(
        should_succeed=True, tool_input="ok_cfg"
    )

    from trulens.feedback import (
        output_schemas as feedback_output_schemas,  # type: ignore[import-not-found]
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=feedback_output_schemas.BaseFeedbackResponse,
    )
    assert out == "ok_cfg"
    assert provider.endpoint.client.responses.create_calls == 1

    # Second call should use cached cfg=True and attempt create again
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}],
        response_format=feedback_output_schemas.BaseFeedbackResponse,
    )
    assert out2 == "ok_cfg"
    assert provider.endpoint.client.responses.create_calls == 2


@pytest.mark.optional
def test_cfg_failure_then_cached_skip(monkeypatch):
    # Use a gpt-5 model so CFG auto-enable is considered
    provider = _make_provider(monkeypatch, model_engine="gpt-5-mini")

    # responses.create will fail; responses.parse also fails -> fallback to chat.completions
    provider.endpoint.client.responses = _DummyResponsesWithCreate(
        should_succeed=False
    )

    from trulens.feedback import (
        output_schemas as feedback_output_schemas,  # type: ignore[import-not-found]
    )

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=feedback_output_schemas.BaseFeedbackResponse,
    )
    # Should have fallen back to chat completions' "ok"
    assert out == "ok"
    assert provider.endpoint.client.responses.create_calls == 1
    # Structured outputs was attempted and failed once, then chat was used
    assert len(provider.endpoint.client.chat.completions.create_calls) >= 1

    # Clear chat call log and call again; cfg should be cached False and skip create
    provider.endpoint.client.chat.completions.create_calls.clear()
    out2 = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi2"}],
        response_format=feedback_output_schemas.BaseFeedbackResponse,
    )
    assert out2 == "ok"
    # Still only 1 attempt to create (from first call)
    assert provider.endpoint.client.responses.create_calls == 1
    # And chat.completions used again
    assert len(provider.endpoint.client.chat.completions.create_calls) == 1
