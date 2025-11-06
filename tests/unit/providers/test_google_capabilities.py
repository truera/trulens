# pyright: reportMissingImports=false, reportMissingModuleSource=false
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    """Ensure each test starts with a clean capability cache"""
    try:
        from trulens.providers.google import Google
    except Exception:
        yield
        return

    Google.clear_model_capabilities_cache()
    yield
    Google.clear_model_capabilities_cache()


class _DummyResponse:
    def __init__(
        self,
        content: str = "ok",
        model_version: str = "gemini-2.5-flash",
        prompt_tokens: int = 100,
        completion_tokens: int = 200,
        reasoning_tokens: int = 0,
    ):
        self.text = content
        self.parsed = None
        self.model_version = model_version

        self.usage_metadata = type(
            "UsageMetadata",
            (),
            {
                "total_token_count": prompt_tokens + completion_tokens,
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": completion_tokens,
                "thoughts_token_count": reasoning_tokens,
            },
        )()

    def to_json_dict(self):
        """Convert to dict format expected by callback - matches real API structure"""
        usage_dict = {
            "total_token_count": self.usage_metadata.total_token_count,
            "prompt_token_count": self.usage_metadata.prompt_token_count,
            "candidates_token_count": self.usage_metadata.candidates_token_count,
            "thoughts_token_count": self.usage_metadata.thoughts_token_count,
        }

        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": self.text}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "model_version": self.model_version,
            "usage_metadata": usage_dict,
        }

    def __str__(self):
        return self.text


class _DummyModels:
    def __init__(self, endpoint=None):
        self.generate_calls: list[Dict[str, Any]] = []
        self.fail_on_params: Dict[str, bool] = {}
        self.response_override: Optional[_DummyResponse] = None
        self.endpoint = endpoint  # Need this to trigger callbacks

    def generate_content(self, *, model, contents, config, **kwargs):
        call_info = {
            "model": model,
            "contents": contents,
            "config_kwargs": config.kwargs if hasattr(config, "kwargs") else {},
        }
        self.generate_calls.append(call_info)

        # Check if any parameter should fail
        config_dict = config.kwargs if hasattr(config, "kwargs") else {}
        for param, should_fail in self.fail_on_params.items():
            if should_fail and param in config_dict:
                raise Exception(f"{param} is not supported")

        # Use override if set, otherwise default response
        response = (
            self.response_override
            if self.response_override
            else _DummyResponse(content="ok")
        )

        # Manually trigger the callback to track usage
        if self.endpoint and hasattr(self.endpoint, "global_callback"):
            response_dict = response.to_json_dict()
            # Check for valid response with usage data
            if (
                "usage_metadata" in response_dict
                and "candidates" in response_dict
            ):
                candidate = response_dict["candidates"][0]
                if candidate.get("finish_reason") == "STOP":
                    self.endpoint.global_callback.handle_generation(
                        response=response_dict
                    )

        return response


def _make_provider(monkeypatch, model_engine: str = "gemini-2.5-flash"):
    """Create a Google provider with mocked client with manual callback tracking"""
    from google.genai import Client
    from trulens.providers.google import Google

    # Create a Mock to mimic the Client class to pass Pydantic validation
    mock_client_instance = Mock(spec=Client)

    # Patch the Client constructor to return mock
    def mock_client_constructor(*args, **kwargs):
        return mock_client_instance

    monkeypatch.setattr("google.genai.Client", mock_client_constructor)

    # Create the provider to use mock client
    provider = Google(
        model_engine=model_engine, api_key="fake-api-key-for-testing"
    )

    # Create DummyModels with endpoint reference to trigger callbacks
    dummy_models = _DummyModels(endpoint=provider.endpoint)
    mock_client_instance.models = dummy_models

    return provider


@pytest.mark.optional
def test_litellm_model_cost_import():
    """Test that litellm model_cost can be imported and has expected structure"""
    try:
        from litellm import model_cost
    except ImportError as e:
        pytest.fail(f"Failed to import model_cost from litellm: {e}")

    assert isinstance(model_cost, dict), "model_cost should be a dictionary"
    assert len(model_cost) > 0, "model_cost should contain pricing data"

    # Check for Google models
    google_models = [
        key for key in model_cost.keys() if "gemini" in key.lower()
    ]
    assert (
        len(google_models) > 0
    ), "model_cost should contain at least one Gemini model"

    # Verify the structure of at least one model entry
    sample_model = next(iter(model_cost.values()))
    assert isinstance(
        sample_model, dict
    ), "Each model entry should be a dictionary"

    # Check for expected pricing fields
    expected_fields = [
        "input_cost_per_token",
        "output_cost_per_token",
        "max_input_tokens",
        "max_output_tokens",
    ]

    # At least one Gemini model should have these basic fields
    has_valid_structure = any(
        all(field in model_cost[google_model] for field in expected_fields)
        for google_model in google_models
    )
    assert (
        has_valid_structure
    ), "At least one Gemini model should have all expected pricing fields"


@pytest.mark.optional
def test_usage_metrics(monkeypatch):
    """Test that reasoning tokens are properly tracked"""
    provider = _make_provider(monkeypatch)

    # Set up a response with reasoning tokens
    dummy_response = _DummyResponse(
        content="test response",
        model_version="gemini-2.5-flash",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=8,
    )
    provider.endpoint.client.models.response_override = dummy_response

    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert out is not None

    # Check that reasoning tokens were tracked
    callback = provider.endpoint.global_callback
    assert callback.cost.n_reasoning_tokens == 8
    assert callback.cost.n_prompt_tokens == 10
    assert callback.cost.n_completion_tokens == 20
    assert callback.cost.n_successful_requests == 1


@pytest.mark.optional
def test_cost_calculation_basic(monkeypatch):
    """Test that cost is properly calculated for a known model"""
    # Mock the litellm model_cost to return real pricing for gemini-2.5-flash
    mock_pricing = {
        "gemini-2.5-flash": {
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 2.5e-06,
            "max_input_tokens": 1048576,
            "max_output_tokens": 65535,
        }
    }
    monkeypatch.setattr(
        "trulens.providers.google.endpoint.model_cost", mock_pricing
    )

    provider = _make_provider(monkeypatch)

    dummy_response = _DummyResponse(
        content="test response",
        model_version="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=2000,
    )
    provider.endpoint.client.models.response_override = dummy_response

    # Call to provider
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert out is not None

    callback = provider.endpoint.global_callback
    expected_cost = 1000 * 3e-07 + 2000 * 2.5e-06
    assert abs(callback.cost.cost - expected_cost) < 1e-10
    assert callback.cost.cost_currency == "USD"


@pytest.mark.optional
def test_multiple_calls_accumulate_costs(monkeypatch):
    """Test that multiple calls accumulate usage and costs"""
    # Mock the litellm model_cost to return real pricing for gemini-2.5-flash
    mock_pricing = {
        "gemini-2.5-flash": {
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 2.5e-06,
            "max_input_tokens": 1048576,
            "max_output_tokens": 65535,
        }
    }
    monkeypatch.setattr(
        "trulens.providers.google.endpoint.model_cost", mock_pricing
    )

    provider = _make_provider(monkeypatch)

    # First call
    dummy_response1 = _DummyResponse(
        content="first response",
        model_version="gemini-2.5-flash",
        prompt_tokens=10,
        completion_tokens=20,
    )
    provider.endpoint.client.models.response_override = dummy_response1

    provider._create_chat_completion(
        messages=[{"role": "user", "content": "first"}],
    )

    # Second call
    dummy_response2 = _DummyResponse(
        content="second response",
        model_version="gemini-2.5-flash",
        prompt_tokens=15,
        completion_tokens=25,
    )
    provider.endpoint.client.models.response_override = dummy_response2

    provider._create_chat_completion(
        messages=[{"role": "user", "content": "second"}],
    )

    # Check accumulated metrics
    callback = provider.endpoint.global_callback
    assert callback.cost.n_prompt_tokens == 25  # 10 + 15
    assert callback.cost.n_completion_tokens == 45  # 20 + 25
    assert callback.cost.n_tokens == 70  # 30 + 40
    assert callback.cost.n_successful_requests == 2

    # Check accumulated cost
    expected_cost = (10 + 15) * 3e-07 + (20 + 25) * 2.5e-06
    assert abs(callback.cost.cost - expected_cost) < 1e-10


@pytest.mark.optional
def test_cost_calculation_unknown_model(monkeypatch):
    """Test that cost calculation handles unknown models gracefully"""
    mock_pricing = {
        "gemini-2.5-flash": {
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 2.5e-06,
            "max_input_tokens": 1048576,
            "max_output_tokens": 65535,
        }
    }
    monkeypatch.setattr(
        "trulens.providers.google.endpoint.model_cost", mock_pricing
    )

    provider = _make_provider(monkeypatch)

    dummy_response = _DummyResponse(
        content="test response",
        model_version="unknown-model",
        prompt_tokens=100,
        completion_tokens=200,
    )
    provider.endpoint.client.models.response_override = dummy_response

    # Call provider with unknown model
    out = provider._create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert out is not None

    # Cost should be 0 for unknown model, but usage should still be tracked
    callback = provider.endpoint.global_callback
    assert callback.cost.cost == 0.0
    assert callback.cost.n_prompt_tokens == 100
    assert callback.cost.n_completion_tokens == 200
