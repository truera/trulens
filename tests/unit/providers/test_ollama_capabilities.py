# pyright: reportMissingImports=false, reportMissingModuleSource=false
from typing import Any, Dict

import httpx
import pydantic
import pytest


class _ParsedModel(pydantic.BaseModel):
    value: str


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> Dict[str, Any]:
        return self._payload


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
    assert provider._native_base_url() == "http://localhost:11434"


@pytest.mark.optional
def test_respects_env_var_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama-host:11434/v1")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="qwen2.5")

    assert provider.model_engine == "qwen2.5"
    assert provider._native_base_url() == "http://my-ollama-host:11434"


@pytest.mark.optional
def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:11434/v1")

    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(base_url="http://explicit-host:11434/v1")

    assert provider._native_base_url() == "http://explicit-host:11434"


@pytest.mark.optional
def test_chat_completion_uses_native_api_and_records_usage(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")

    captured: Dict[str, Any] = {}

    def fake_post(*, url, json, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({
            "message": {"role": "assistant", "content": "hello there"},
            "prompt_eval_count": 12,
            "eval_count": 4,
        })

    monkeypatch.setattr(httpx, "post", fake_post)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert out == "hello there"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llama3.2"
    assert captured["json"]["stream"] is False
    assert "options" not in captured["json"]
    assert "keep_alive" not in captured["json"]

    cost = provider.endpoint.global_callback.cost
    assert cost.n_prompt_tokens == 12
    assert cost.n_completion_tokens == 4
    assert cost.n_tokens == 16


@pytest.mark.optional
def test_options_and_keep_alive_are_forwarded_and_mergeable(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(
        model_engine="llama3.2",
        options={"num_ctx": 4096},
        keep_alive="5m",
    )

    captured: Dict[str, Any] = {}

    def fake_post(*, url, json, timeout):  # noqa: ANN001
        captured["json"] = json
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(httpx, "post", fake_post)

    # Per-call `options` should merge on top of the provider-level default,
    # and a per-call `keep_alive` should override the provider-level one.
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        options={"repeat_penalty": 1.2},
        keep_alive="1m",
    )

    assert out == "ok"
    assert captured["json"]["options"] == {
        "num_ctx": 4096,
        "repeat_penalty": 1.2,
    }
    assert captured["json"]["keep_alive"] == "1m"


@pytest.mark.optional
def test_temperature_and_seed_move_into_options(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")

    captured: Dict[str, Any] = {}

    def fake_post(*, url, json, timeout):  # noqa: ANN001
        captured["json"] = json
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        seed=123,
        reasoning_effort="medium",
    )

    assert captured["json"]["options"] == {"temperature": 0.0, "seed": 123}
    # Ollama has no native reasoning_effort control; it must not leak
    # through as a top-level (invalid) request field.
    assert "reasoning_effort" not in captured["json"]


@pytest.mark.optional
def test_response_format_becomes_native_json_schema(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")

    captured: Dict[str, Any] = {}

    def fake_post(*, url, json, timeout):  # noqa: ANN001
        captured["json"] = json
        return _FakeResponse({"message": {"content": '{"value": "ok"}'}})

    monkeypatch.setattr(httpx, "post", fake_post)

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_ParsedModel,
    )

    assert out == '{"value": "ok"}'
    assert captured["json"]["format"] == _ParsedModel.model_json_schema()


@pytest.mark.optional
def test_list_models(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama()

    def fake_get(url, timeout):  # noqa: ANN001
        assert url == "http://localhost:11434/api/tags"
        return _FakeResponse({
            "models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]
        })

    monkeypatch.setattr(httpx, "get", fake_get)

    assert provider.list_models() == ["llama3.2:latest", "qwen2.5:7b"]


@pytest.mark.optional
def test_pull_model_defaults_to_model_engine(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")

    captured: Dict[str, Any] = {}

    def fake_post(url, *, json, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"status": "success"})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider.pull_model()

    assert captured["url"] == "http://localhost:11434/api/pull"
    assert captured["json"] == {"name": "llama3.2", "stream": False}


@pytest.mark.optional
def test_pull_model_raises_on_failure_status(monkeypatch):
    from trulens.providers.ollama import (
        Ollama,  # type: ignore[import-not-found]
    )

    provider = Ollama(model_engine="llama3.2")

    def fake_post(url, *, json, timeout):  # noqa: ANN001
        return _FakeResponse({"status": "error", "error": "not found"})

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError):
        provider.pull_model("nonexistent-model")
