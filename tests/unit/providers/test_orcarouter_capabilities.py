# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pytest


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

    # Structured-output and moderation support are inherited unchanged from
    # the OpenAI provider, since OrcaRouter is API-compatible with OpenAI.
    assert isinstance(provider, OpenAI)
    assert provider._structured_output_supported() is True
