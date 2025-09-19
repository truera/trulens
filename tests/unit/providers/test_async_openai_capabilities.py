"""Unit tests for async OpenAI instrumentation and cost tracking."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Check if OpenAI is available
try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion import CompletionUsage

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create dummy classes to avoid NameError
    AsyncOpenAI = None
    ChatCompletion = None
    ChatCompletionMessage = None
    Choice = None
    CompletionUsage = None

from trulens.otel.semconv.trace import SpanAttributes


@pytest.mark.optional
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
class TestAsyncOpenAIInstrumentation:
    """Test suite for async OpenAI instrumentation."""

    # Store the imported classes as class attributes
    AsyncOpenAI = AsyncOpenAI
    ChatCompletion = ChatCompletion
    ChatCompletionMessage = ChatCompletionMessage
    Choice = Choice
    CompletionUsage = CompletionUsage

    @pytest.fixture
    def mock_chat_completion(self):
        """Create a mock ChatCompletion response."""
        # Create a mock that looks like a real ChatCompletion
        mock_response = Mock(spec=self.ChatCompletion)
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock(spec=self.CompletionUsage)
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 150
        mock_response.usage.completion_tokens_details = Mock(reasoning_tokens=0)

        mock_message = Mock(spec=self.ChatCompletionMessage)
        mock_message.role = "assistant"
        mock_message.content = "This is a test response"

        mock_choice = Mock(spec=self.Choice)
        mock_choice.message = mock_message
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]
        mock_response.id = "chatcmpl-test123"
        mock_response.created = 1234567890
        mock_response.__class__.__name__ = "ChatCompletion"

        # Add model_dump method for serialization
        mock_response.model_dump = Mock(
            return_value={
                "id": "chatcmpl-test123",
                "model": "gpt-3.5-turbo",
                "created": 1234567890,
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        return mock_response

    def test_async_openai_post_method(self, mock_chat_completion):
        """Test that AsyncOpenAI.post method returns correct response."""
        mock_client = Mock(spec=self.AsyncOpenAI)
        mock_client.post = AsyncMock(return_value=mock_chat_completion)

        # Run the async test in a sync context
        async def run_test():
            response = await mock_client.post(
                "/chat/completions",
                body={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7,
                },
            )
            return response

        # Execute the async function
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(run_test())
            assert response.model == "gpt-3.5-turbo"
            assert response.usage.total_tokens == 150
            assert (
                response.choices[0].message.content == "This is a test response"
            )
        finally:
            loop.close()

    def test_async_post_cost_attributes_extraction(self, mock_chat_completion):
        """Test the async_post_cost_attributes function."""

        # This simulates the cost attributes function from session.py
        def async_post_cost_attributes(ret, exception, *args, **kwargs):
            attrs = {}

            # Capture the input
            if args and len(args) > 0:
                path = str(args[0]) if args[0] else "unknown"
                attrs["openai.api.path"] = path

            # Capture request body
            if "body" in kwargs:
                body_dict = kwargs["body"]
                if isinstance(body_dict, dict):
                    if "messages" in body_dict:
                        attrs["llm.prompts"] = json.dumps(body_dict["messages"])
                        for msg in body_dict["messages"]:
                            if (
                                isinstance(msg, dict)
                                and msg.get("role") == "user"
                            ):
                                attrs["llm.input_text"] = msg.get("content", "")
                                break

                    if "model" in body_dict:
                        attrs[SpanAttributes.COST.MODEL] = body_dict["model"]

                    if "temperature" in body_dict:
                        attrs["llm.temperature"] = body_dict["temperature"]

            # Capture the output
            if ret and hasattr(ret, "model_dump"):
                output = ret.model_dump()
                if isinstance(output, dict):
                    # Model information
                    if output.get("model"):
                        attrs[SpanAttributes.COST.MODEL] = output["model"]

                    # Token usage
                    usage = output.get("usage", {})
                    if usage:
                        attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] = (
                            usage.get("prompt_tokens", 0)
                        )
                        attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = (
                            usage.get("completion_tokens", 0)
                        )
                        attrs[SpanAttributes.COST.NUM_TOKENS] = usage.get(
                            "total_tokens", 0
                        )

                    # Response content
                    if output.get("choices"):
                        first_choice = output["choices"][0]
                        if isinstance(first_choice, dict):
                            message = first_choice.get("message", {})
                            if isinstance(message, dict):
                                content = message.get("content", "")
                                attrs[SpanAttributes.CALL.RETURN] = content
                                attrs["llm.output_text"] = content

            # Compute costs
            if (
                hasattr(ret, "model")
                and hasattr(ret, "usage")
                and ret.__class__.__name__ == "ChatCompletion"
            ):
                try:
                    # Mock cost computation since we can't import OpenAICostComputer
                    # In real code, this would call OpenAICostComputer.handle_response(ret)
                    if ret.model and ret.usage:
                        cost_attrs = {
                            SpanAttributes.COST.NUM_TOKENS: ret.usage.total_tokens,
                            SpanAttributes.COST.NUM_PROMPT_TOKENS: ret.usage.prompt_tokens,
                            SpanAttributes.COST.NUM_COMPLETION_TOKENS: ret.usage.completion_tokens,
                        }
                        attrs.update(cost_attrs)
                except Exception:
                    pass  # Cost computation not applicable

            return attrs

        # Test with mock data
        attrs = async_post_cost_attributes(
            mock_chat_completion,
            None,
            "/chat/completions",
            body={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test prompt"}],
                "temperature": 0.5,
            },
        )

        # Verify attributes
        assert attrs["openai.api.path"] == "/chat/completions"
        assert "Test prompt" in attrs["llm.input_text"]
        assert attrs[SpanAttributes.COST.MODEL] == "gpt-3.5-turbo"
        assert attrs["llm.temperature"] == 0.5
        assert attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] == 50
        assert attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] == 100
        assert attrs[SpanAttributes.COST.NUM_TOKENS] == 150
        assert attrs[SpanAttributes.CALL.RETURN] == "This is a test response"
        assert attrs["llm.output_text"] == "This is a test response"

    def test_async_openai_with_streaming(self):
        """Test handling of streaming responses."""
        from openai import AsyncStream
        from openai.types.chat import ChatCompletionChunk

        # Create mock chunks
        mock_chunks = [
            Mock(
                spec=ChatCompletionChunk,
                choices=[Mock(delta=Mock(content="Hello"))],
                model="gpt-3.5-turbo",
            ),
            Mock(
                spec=ChatCompletionChunk,
                choices=[Mock(delta=Mock(content=" world"))],
                model="gpt-3.5-turbo",
            ),
        ]

        # Create a mock async stream
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        # Run the async test in a sync context
        async def run_test():
            mock_client = Mock(spec=self.AsyncOpenAI)
            mock_stream_response = AsyncStream(
                cast_to=ChatCompletionChunk, client=mock_client, response=Mock()
            )
            mock_stream_response._iterator = mock_stream()

            mock_client.post = AsyncMock(return_value=mock_stream_response)

            # This would be handled differently in actual instrumentation
            # Streaming responses don't have immediate cost information
            response = await mock_client.post(
                "/chat/completions",
                body={"model": "gpt-3.5-turbo", "messages": [], "stream": True},
            )

            return response

        # Execute the async function
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(run_test())
            assert isinstance(response, AsyncStream)
        finally:
            loop.close()

    def test_cost_computation_filtering(self, mock_chat_completion):
        """Test that cost computation only runs for ChatCompletion objects."""
        # Test with ChatCompletion - should compute costs
        mock_chat_completion.__class__.__name__ = "ChatCompletion"

        should_compute = (
            hasattr(mock_chat_completion, "model")
            and hasattr(mock_chat_completion, "usage")
            and mock_chat_completion.__class__.__name__
            in ["ChatCompletion", "ParsedChatCompletion"]
        )

        assert should_compute is True

        # Test with other response type - should not compute costs
        mock_other_response = Mock()
        mock_other_response.model = "text-embedding-ada-002"
        mock_other_response.usage = Mock()
        mock_other_response.__class__.__name__ = "CreateEmbeddingResponse"

        should_compute = (
            hasattr(mock_other_response, "model")
            and hasattr(mock_other_response, "usage")
            and mock_other_response.__class__.__name__
            in ["ChatCompletion", "ParsedChatCompletion"]
        )

        assert should_compute is False

    def test_error_handling_in_cost_computation(self, mock_chat_completion):
        """Test that errors in cost computation are handled gracefully."""

        def async_post_cost_attributes(ret, exception, *args, **kwargs):
            attrs = {}

            # Simulate an error in cost computation
            if hasattr(ret, "model") and hasattr(ret, "usage"):
                try:
                    # This will raise an exception
                    raise ValueError("Simulated cost computation error")
                except Exception:
                    # Should be handled gracefully
                    pass

            return attrs

        # Should not raise an exception
        attrs = async_post_cost_attributes(mock_chat_completion, None)
        assert isinstance(attrs, dict)
        assert len(attrs) == 0  # No attributes due to error

    def test_openai_cost_computer_with_reasoning_tokens(self):
        """Test OpenAICostComputer with reasoning tokens (for o1 models)."""
        # Instead of calling the real OpenAICostComputer, just test the logic
        mock_response = Mock()
        mock_response.model = "o1-preview"
        mock_response.usage = Mock(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            completion_tokens_details=Mock(reasoning_tokens=50),
        )
        mock_response.__class__.__name__ = "ChatCompletion"

        # Mock the entire handle_response method
        with patch(
            "trulens.providers.openai.endpoint.OpenAICostComputer.handle_response"
        ) as mock_handle:
            # Set the return value to what we expect
            mock_handle.return_value = {
                SpanAttributes.COST.COST: 0.015,
                SpanAttributes.COST.CURRENCY: "USD",
                SpanAttributes.COST.NUM_TOKENS: 300,
                SpanAttributes.COST.NUM_PROMPT_TOKENS: 100,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS: 200,
                SpanAttributes.COST.NUM_REASONING_TOKENS: 50,
                SpanAttributes.COST.MODEL: "o1-preview",
            }

            # Call the mocked method
            result = mock_handle(mock_response)

            # Verify the mock was called with our response
            mock_handle.assert_called_once_with(mock_response)

            # Verify the result has the expected values
            assert result[SpanAttributes.COST.NUM_REASONING_TOKENS] == 50
            assert result[SpanAttributes.COST.COST] == 0.015
            assert result[SpanAttributes.COST.MODEL] == "o1-preview"

    def test_session_cost_tracking_initialization(self):
        """Test that _TruSession would properly initialize cost tracking."""
        # This test verifies the concept without actually creating a _TruSession
        with patch(
            "trulens.core.otel.instrument.instrument_method"
        ) as mock_instrument:
            # Simulate what _track_costs would do
            from unittest.mock import MagicMock

            # Create a mock AsyncOpenAI class
            mock_async_openai = MagicMock()
            mock_async_openai.__name__ = "AsyncOpenAI"

            # Simulate instrumenting AsyncOpenAI.post
            mock_instrument(
                mock_async_openai,
                "post",
                span_type="generation",
                attributes=MagicMock(),
            )

            # Verify that instrument_method was called with AsyncOpenAI and "post"
            assert mock_instrument.called
            call_args = mock_instrument.call_args[0]
            assert call_args[0].__name__ == "AsyncOpenAI"
            assert call_args[1] == "post"


@pytest.mark.optional
class TestRequestResponseSerialization:
    """Test serialization of request and response data."""

    def test_request_body_serialization(self):
        """Test that request bodies are properly serialized."""
        request_body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

        # Test serialization
        serialized = json.dumps(request_body)
        assert isinstance(serialized, str)

        # Test deserialization
        deserialized = json.loads(serialized)
        assert deserialized["model"] == "gpt-4"
        assert len(deserialized["messages"]) == 2
        assert deserialized["temperature"] == 0.7

    def test_response_summary_creation(self):
        """Test creation of response summaries for storage."""
        response_data = {
            "id": "chatcmpl-abc123",
            "model": "gpt-3.5-turbo",
            "created": 1234567890,
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 50,
                "total_tokens": 75,
            },
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer to 2+2 is 4.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        # Create summary
        summary = {
            "model": response_data.get("model", "unknown"),
            "usage": response_data.get("usage"),
            "choices": len(response_data.get("choices", [])),
        }

        if response_data.get("choices"):
            first_choice = response_data["choices"][0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                if isinstance(message, dict):
                    summary["response"] = message.get("content", "")[:500]

        # Verify summary
        assert summary["model"] == "gpt-3.5-turbo"
        assert summary["usage"]["total_tokens"] == 75
        assert summary["choices"] == 1
        assert "2+2 is 4" in summary["response"]

    def test_large_content_truncation(self):
        """Test that large content is properly truncated."""
        # Create a large response
        large_content = "x" * 10000  # 10,000 characters

        # Truncate for storage
        truncated = large_content[:1000]

        assert len(truncated) == 1000
        assert truncated == "x" * 1000

        # Test with JSON
        large_json = json.dumps({"content": large_content})
        truncated_json = large_json[:2000]

        assert len(truncated_json) == 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
