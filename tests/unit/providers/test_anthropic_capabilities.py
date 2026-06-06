# Tests for Anthropic provider capabilities
import types
from typing import Any, Dict, List, Optional

import pydantic
import pytest


@pytest.fixture(autouse=True)
def _reset_model_capabilities_cache():
    """Ensure each test starts with a clean capability cache."""
    from trulens.providers.anthropic import Anthropic

    Anthropic.clear_model_capabilities_cache()
    yield
    Anthropic.clear_model_capabilities_cache()


class _ParsedModel(pydantic.BaseModel):
    value: str


class _DummyContentBlock:
    """Simulates an Anthropic content block."""

    def __init__(self, block_type: str, text: str = "", tool_input: dict = None):
        self.type = block_type
        self.text = text
        self.input = tool_input or {}


class _DummyUsage:
    def __init__(
        self,
        input_tokens: int = 10,
        output_tokens: int = 20,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _DummyResponse:
    """Simulates an Anthropic Message response."""

    def __init__(
        self,
        content: List[_DummyContentBlock],
        model: str = "claude-sonnet-4-6",
        usage: Optional[_DummyUsage] = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or _DummyUsage()


class _DummyMessages:
    """Simulates anthropic.Anthropic.messages with a create method."""

    def __init__(self, response: _DummyResponse):
        self._response = response
        self.create_calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._response


class _DummyClient:
    """Simulates an anthropic.Anthropic client."""

    def __init__(self, messages: _DummyMessages):
        self.messages = messages


class TestAnthropicMessagesConversion:
    """Test the OpenAI → Anthropic message format conversion."""

    def test_system_extraction(self):
        """System messages should be extracted as a top-level string."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": "Evaluate this text."},
        ]
        system, remaining = Anthropic._extract_system_from_messages(messages)
        assert system == "You are a helpful judge."
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"

    def test_multiple_system_messages_merged(self):
        """Multiple system messages should be merged with newlines."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hello"},
        ]
        system, remaining = Anthropic._extract_system_from_messages(messages)
        assert "Rule 1." in system
        assert "Rule 2." in system
        assert len(remaining) == 1

    def test_no_system_message(self):
        """No system messages: system should be empty string."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        system, remaining = Anthropic._extract_system_from_messages(messages)
        assert system == ""
        assert len(remaining) == 2

    def test_user_to_anthropic_format(self):
        """User messages should be converted to Anthropic content blocks."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [{"role": "user", "content": "Hello world"}]
        result = Anthropic._convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello world"}]

    def test_assistant_to_anthropic_format(self):
        """Assistant messages should be converted to Anthropic content blocks."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [{"role": "assistant", "content": "I can help"}]
        result = Anthropic._convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "I can help"}]

    def test_consecutive_same_role_merged(self):
        """Consecutive same-role messages should be merged (Anthropic req)."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
        ]
        result = Anthropic._convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["text"] == "Part 1"
        assert result[0]["content"][1]["text"] == "Part 2"

    def test_tool_role_conversion(self):
        """Tool messages should become user messages with tool_result blocks."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {"role": "assistant", "content": "Let me check"},
            {
                "role": "tool",
                "content": '{"result": 42}',
                "tool_call_id": "call_123",
            },
        ]
        result = Anthropic._convert_messages_to_anthropic(messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "call_123"

    def test_list_content_passthrough(self):
        """List-format content should pass through unchanged."""
        from trulens.providers.anthropic.provider import Anthropic

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "source": {}},
                ],
            }
        ]
        result = Anthropic._convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["content"] == messages[0]["content"]


class TestAnthropicCreateChatCompletion:
    """Test the _create_chat_completion method with mocked client."""

    def test_basic_text_completion(self):
        """Basic text completion returns the response text."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "This is a test response.")]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        # Monkey-patch the endpoint to use our dummy client
        provider = Anthropic(model_engine="claude-sonnet-4-6")
        provider.endpoint.client.client = dummy_client

        result = provider._create_chat_completion(
            prompt="Hello, Claude!",
        )

        assert result == "This is a test response."
        assert len(dummy_messages.create_calls) == 1
        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert len(call_kwargs["messages"]) == 1

    def test_messages_input(self):
        """Messages-based input should be converted correctly."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "Score: 8")]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        result = provider._create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Rate this: good"},
            ]
        )

        assert result == "Score: 8"
        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["system"] == "You are a judge."
        assert len(call_kwargs["messages"]) == 1  # Only user message
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_structured_output_via_tool_use(self):
        """Structured output should use tool_use and return parsed model."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[
                _DummyContentBlock(
                    "tool_use",
                    tool_input={"value": "parsed_result"},
                )
            ]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        result = provider._create_chat_completion(
            prompt="Parse this",
            response_format=_ParsedModel,
        )

        assert isinstance(result, _ParsedModel)
        assert result.value == "parsed_result"

        call_kwargs = dummy_messages.create_calls[0]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "output_format"
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "output_format",
        }

    def test_fallback_to_text_when_tool_use_fails(self):
        """When tool_use parsing fails, fall back to text extraction."""
        from trulens.providers.anthropic.provider import Anthropic

        # Response has tool_use with invalid input AND a text block
        dummy_response = _DummyResponse(
            content=[
                _DummyContentBlock(
                    "tool_use",
                    tool_input={"wrong_field": "bad"},
                ),
                _DummyContentBlock("text", "Fallback text response"),
            ]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        result = provider._create_chat_completion(
            prompt="Parse this",
            response_format=_ParsedModel,  # Requires "value" field
        )

        # Should fall back to text since tool_use.input has "wrong_field"
        assert result == "Fallback text response"

    def test_missing_prompt_and_messages_raises(self):
        """Should raise ValueError if neither prompt nor messages provided."""
        from trulens.providers.anthropic.provider import Anthropic

        provider = Anthropic()
        with pytest.raises(ValueError, match="prompt.*messages"):
            provider._create_chat_completion()

    def test_model_kwarg_override(self):
        """The model kwarg should override the default model_engine."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "ok")],
            model="claude-opus-4-8",
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic(model_engine="claude-sonnet-4-6")
        provider.endpoint.client.client = dummy_client

        provider._create_chat_completion(
            prompt="test",
            model="claude-opus-4-8",
        )

        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["model"] == "claude-opus-4-8"

    def test_temperature_default(self):
        """Non-reasoning models should default to temperature=0.0."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "ok")]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        provider._create_chat_completion(prompt="test")

        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["temperature"] == 0.0

    def test_temperature_explicit(self):
        """Explicit temperature should override default."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "ok")]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        provider._create_chat_completion(prompt="test", temperature=0.7)

        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["temperature"] == 0.7

    def test_max_tokens_default(self):
        """Default max_tokens should be 4096."""
        from trulens.providers.anthropic.provider import Anthropic

        dummy_response = _DummyResponse(
            content=[_DummyContentBlock("text", "ok")]
        )
        dummy_messages = _DummyMessages(dummy_response)
        dummy_client = _DummyClient(dummy_messages)

        provider = Anthropic()
        provider.endpoint.client.client = dummy_client

        provider._create_chat_completion(prompt="test")

        call_kwargs = dummy_messages.create_calls[0]
        assert call_kwargs["max_tokens"] == 4096


class TestAnthropicProviderInit:
    """Test Anthropic provider initialization."""

    def test_default_model_engine(self):
        """Default model should be claude-sonnet-4-6."""
        from trulens.providers.anthropic.provider import Anthropic

        provider = Anthropic()
        assert provider.model_engine == "claude-sonnet-4-6"

    def test_custom_model_engine(self):
        """Custom model_engine should override the default."""
        from trulens.providers.anthropic.provider import Anthropic

        provider = Anthropic(model_engine="claude-opus-4-8")
        assert provider.model_engine == "claude-opus-4-8"


class TestAnthropicPricing:
    """Test Anthropic cost computation."""

    def test_known_model_pricing(self):
        """Known models should have correct pricing."""
        from trulens.providers.anthropic.endpoint import _get_model_pricing

        in_p, out_p = _get_model_pricing("claude-sonnet-4-6")
        assert in_p == 3.0
        assert out_p == 15.0

        in_p, out_p = _get_model_pricing("claude-opus-4-8")
        assert in_p == 15.0
        assert out_p == 75.0

        in_p, out_p = _get_model_pricing("claude-haiku-4-5")
        assert in_p == 0.80
        assert out_p == 4.0

    def test_unknown_model_fallback(self):
        """Unknown models should use default pricing."""
        from trulens.providers.anthropic.endpoint import _get_model_pricing

        in_p, out_p = _get_model_pricing("some-future-model")
        assert in_p == 3.0
        assert out_p == 15.0

    def test_cost_computation(self):
        """Cost should be computed correctly from usage."""
        from trulens.providers.anthropic.endpoint import AnthropicCostComputer

        response = _DummyResponse(
            content=[_DummyContentBlock("text", "Hello")],
            model="claude-sonnet-4-6",
            usage=_DummyUsage(input_tokens=1000, output_tokens=500),
        )

        result = AnthropicCostComputer.handle_response(response)
        # 1000/1M * 3.0 + 500/1M * 15.0 = 0.003 + 0.0075 = 0.0105
        expected_cost = (1000 / 1e6) * 3.0 + (500 / 1e6) * 15.0
        assert result["cost"] == pytest.approx(expected_cost)
        assert result["n_tokens"] == 1500
        assert result["n_prompt_tokens"] == 1000
        assert result["n_completion_tokens"] == 500
        assert result["model"] == "claude-sonnet-4-6"
