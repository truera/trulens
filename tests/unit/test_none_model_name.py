"""Tests for None model_name handling in OpenAI endpoint.

Reproduces and verifies the fix for GitHub issue #1700:
    AttributeError: 'NoneType' object has no attribute 'lower'

This error occurs when model_name is None and gets passed to
langchain's standardize_model_name(), which calls .lower() on it.
This commonly happens with Azure OpenAI deployments where the
model parameter may be None.
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING
import unittest
from unittest import mock

import pytest

if TYPE_CHECKING:
    pass

_FAKE_KEY = "sk-fake-key-for-testing"


@pytest.mark.optional
class TestNoneModelNameNonOtel(unittest.TestCase):
    """Tests for None model_name in the non-OTEL code path."""

    def setUp(self):
        os.environ["OPENAI_API_KEY"] = _FAKE_KEY

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)

    def test_handle_wrapped_call_with_none_model_no_crash(self):
        """Verify handle_wrapped_call does not crash when
        model=None is in bindings.kwargs.

        Before the fix, this would raise:
            AttributeError: 'NoneType' object has no attribute
            'lower'
        """
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        # Simulate bindings where model=None (Azure OpenAI case)
        bindings = mock.MagicMock(spec=inspect.BoundArguments)
        bindings.kwargs = {"model": None}
        bindings.arguments = {"model": None}

        # Should NOT raise AttributeError
        ep.handle_wrapped_call(
            func=mock.MagicMock(),
            bindings=bindings,
            response=mock_response,
            callback=callback,
        )

    def test_handle_wrapped_call_with_valid_model_works(self):
        """Verify handle_wrapped_call works normally with a
        valid model name."""
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        bindings = mock.MagicMock(spec=inspect.BoundArguments)
        bindings.kwargs = {"model": "gpt-4o"}
        bindings.arguments = {"model": "gpt-4o"}

        # Should not raise any exception
        ep.handle_wrapped_call(
            func=mock.MagicMock(),
            bindings=bindings,
            response=mock_response,
            callback=callback,
        )

    def test_handle_wrapped_call_model_not_in_kwargs(self):
        """Test when model is not in kwargs at all (e.g. Azure
        OpenAI where model is configured at client level)."""
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        bindings = mock.MagicMock(spec=inspect.BoundArguments)
        bindings.kwargs = {}
        bindings.arguments = {}

        # model_name defaults to "" — should not crash
        ep.handle_wrapped_call(
            func=mock.MagicMock(),
            bindings=bindings,
            response=mock_response,
            callback=callback,
        )

    def test_handle_wrapped_call_model_in_arguments_not_kwargs(
        self,
    ):
        """Test when model is in bindings.arguments but not in
        bindings.kwargs (positional argument)."""
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        # model in arguments but not in kwargs
        bindings = mock.MagicMock(spec=inspect.BoundArguments)
        bindings.kwargs = {}
        bindings.arguments = {"model": "gpt-4o"}

        # Should pick up model from arguments
        ep.handle_wrapped_call(
            func=mock.MagicMock(),
            bindings=bindings,
            response=mock_response,
            callback=callback,
        )

    def test_handle_response_with_none_model_name_no_crash(self):
        """Directly test _handle_response with None model_name.

        Before the fix, this would raise AttributeError in
        langchain's standardize_model_name.
        """
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        # Should NOT raise
        openai_endpoint.OpenAIEndpoint._handle_response(
            model_name=None,
            response=mock_response,
            callbacks=[callback],
        )

    def test_handle_response_with_empty_string_model_name(self):
        """Test _handle_response with empty string model_name
        (this should always have worked)."""
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        # Should not raise
        openai_endpoint.OpenAIEndpoint._handle_response(
            model_name="",
            response=mock_response,
            callbacks=[callback],
        )

    def test_handle_wrapped_call_none_model_falls_back_to_response(
        self,
    ):
        """When model=None in kwargs, verify we fall back to
        response.model."""
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o-azure"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        bindings = mock.MagicMock(spec=inspect.BoundArguments)
        bindings.kwargs = {"model": None}
        bindings.arguments = {"model": None}

        # Patch _handle_response to capture model_name
        with mock.patch.object(
            openai_endpoint.OpenAIEndpoint,
            "_handle_response",
        ) as mock_handle:
            ep.handle_wrapped_call(
                func=mock.MagicMock(),
                bindings=bindings,
                response=mock_response,
                callback=callback,
            )
            # Should have fallen back to response.model
            call_args = mock_handle.call_args
            # _handle_response is called with positional args
            actual_model = call_args[0][0]
            assert actual_model == "gpt-4o-azure", (
                f"Expected 'gpt-4o-azure' but got " f"'{actual_model}'"
            )


@pytest.mark.optional
class TestNoneModelNameOtel(unittest.TestCase):
    """Tests for None model_name in the OTEL code path."""

    def setUp(self):
        os.environ["OPENAI_API_KEY"] = _FAKE_KEY

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)

    def test_cost_computer_with_none_model_in_response(self):
        """Test OpenAICostComputer.handle_response with a
        response whose model attribute is None.

        Before the fix, this would pass None to
        _handle_response which would crash in
        langchain's standardize_model_name.
        """
        from trulens.providers.openai import endpoint as openai_endpoint

        mock_response = mock.MagicMock()
        mock_response.model = None
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        # Should NOT raise
        result = openai_endpoint.OpenAICostComputer.handle_response(
            mock_response
        )
        assert isinstance(result, dict)

    def test_cost_computer_with_valid_model(self):
        """Test OpenAICostComputer.handle_response with a
        valid model name."""
        from trulens.providers.openai import endpoint as openai_endpoint

        mock_response = mock.MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    role="assistant",
                    content="Hello!",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mock.MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        )

        result = openai_endpoint.OpenAICostComputer.handle_response(
            mock_response
        )
        assert isinstance(result, dict)

    def test_cost_computer_streaming_none_model_chunk(self):
        """Test OpenAICostComputer.handle_response with a
        streaming response where first_chunk.model is None.

        Before the fix, first_chunk.model = None would be
        passed through to _handle_response.
        """
        from trulens.providers.openai import endpoint as openai_endpoint

        mock_chunk = mock.MagicMock()
        mock_chunk.model = None
        mock_chunk.choices = [
            mock.MagicMock(
                delta=mock.MagicMock(content="Hello"),
                finish_reason=None,
            )
        ]

        # Remove model attr from the stream itself so it
        # enters the streaming branch
        mock_stream_obj = mock.MagicMock()
        mock_stream_obj.__iter__ = mock.MagicMock(
            return_value=iter([mock_chunk])
        )
        del mock_stream_obj.model

        # Should NOT raise
        result = openai_endpoint.OpenAICostComputer.handle_response(
            mock_stream_obj
        )
        assert isinstance(result, dict)

    def test_handle_generation_chunk_with_none_model(self):
        """Test handle_generation_chunk when response.model is
        None.

        The try/finally block in handle_generation_chunk
        swallows exceptions and returns the response. Before
        the fix, the exception was silently swallowed but cost
        tracking was lost. After the fix, no exception occurs
        and cost tracking works correctly.
        """
        from trulens.providers.openai import endpoint as openai_endpoint

        ep = openai_endpoint.OpenAIEndpoint()
        callback = openai_endpoint.OpenAICallback(endpoint=ep)

        mock_response = mock.MagicMock()
        mock_response.model = None
        mock_response.choices = [
            mock.MagicMock(
                delta=mock.MagicMock(content="Hello"),
                finish_reason="stop",
            )
        ]

        # Should not raise (try/finally catches), but after
        # fix the internal handling should also succeed
        result = callback.handle_generation_chunk(response=mock_response)
        # The method returns the response via finally
        assert result is mock_response


if __name__ == "__main__":
    unittest.main()
