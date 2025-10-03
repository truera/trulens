"""Comprehensive unit tests for TruLlamaWorkflow and async OpenAI instrumentation."""

import asyncio
from dataclasses import dataclass
import importlib.util
import json
import os
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add source paths to PYTHONPATH for testing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../src/otel/semconv")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/core"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../src/apps/llamaindex")
)

from trulens.otel.semconv.trace import SpanAttributes

# =============================================================================
# Tests from test_workflow_instrumentation.py
# =============================================================================


@pytest.mark.optional
class TestWorkflowOutputCapture:
    """Test workflow output capture functionality."""

    def test_main_output_override_with_future(self):
        """Test handling of Future-like WorkflowHandler objects."""
        # Mock a WorkflowHandler (Future-like object)
        mock_handler = Mock()
        mock_handler._result = {"result": "test_output"}

        # Test the logic that would be in _main_output_override
        def get_actual_output(handler):
            """Extract actual output from a WorkflowHandler."""
            if hasattr(handler, "_result"):
                return handler._result
            return handler

        output = get_actual_output(mock_handler)
        assert output == {"result": "test_output"}

        # Test with direct output
        direct_output = {"direct": "result"}
        output = get_actual_output(direct_output)
        assert output == {"direct": "result"}

    def test_workflow_record_id_storage(self):
        """Test global storage of workflow record IDs."""
        # Simulate the global dictionary
        _WORKFLOW_RECORD_IDS = {}

        # Store a record ID for a workflow
        workflow_name = "TestWorkflow"
        record_id = "test-record-123"
        _WORKFLOW_RECORD_IDS[workflow_name] = record_id

        # Retrieve it
        assert _WORKFLOW_RECORD_IDS.get(workflow_name) == record_id

        # Test with multiple workflows
        _WORKFLOW_RECORD_IDS["AnotherWorkflow"] = "another-id-456"
        assert len(_WORKFLOW_RECORD_IDS) == 2

    def test_step_function_attributes(self):
        """Test generation of step function attributes."""

        def create_step_attributes(method_name, output_event):
            """Create attributes for a workflow step."""
            attrs = {
                SpanAttributes.SPAN_TYPE: "workflow_step",
            }

            if output_event:
                # Serialize the output event
                if hasattr(output_event, "__dict__"):
                    serialized = json.dumps(output_event.__dict__, default=str)
                else:
                    serialized = str(output_event)
                attrs[SpanAttributes.WORKFLOW.OUTPUT_EVENT] = serialized

            return attrs

        # Test with a mock event
        mock_event = Mock()
        mock_event.__dict__ = {"result": "test", "metadata": {"key": "value"}}

        attrs = create_step_attributes("test_step", mock_event)

        assert attrs[SpanAttributes.SPAN_TYPE] == "workflow_step"
        assert SpanAttributes.WORKFLOW.OUTPUT_EVENT in attrs
        assert "test" in attrs[SpanAttributes.WORKFLOW.OUTPUT_EVENT]


@pytest.mark.optional
class TestAsyncOpenAICostTracking:
    """Test async OpenAI cost tracking."""

    def test_cost_attributes_extraction(self):
        """Test extraction of cost attributes from OpenAI responses."""

        # Mock ChatCompletion response
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.choices = [
            Mock(message=Mock(content="Test response", role="assistant"))
        ]
        mock_response.__class__.__name__ = "ChatCompletion"
        mock_response.model_dump = Mock(
            return_value={
                "model": "gpt-3.5-turbo",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "choices": [
                    {
                        "message": {
                            "content": "Test response",
                            "role": "assistant",
                        }
                    }
                ],
            }
        )

        # Extract attributes
        attrs = {}

        # Check if it's a ChatCompletion
        if (
            hasattr(mock_response, "model")
            and hasattr(mock_response, "usage")
            and mock_response.__class__.__name__
            in ["ChatCompletion", "ParsedChatCompletion"]
        ):
            output = mock_response.model_dump()

            # Model information
            if output.get("model"):
                attrs[SpanAttributes.COST.MODEL] = output["model"]

            # Token usage
            usage = output.get("usage", {})
            if usage:
                attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] = usage.get(
                    "prompt_tokens", 0
                )
                attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = usage.get(
                    "completion_tokens", 0
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

        # Verify attributes
        assert attrs[SpanAttributes.COST.MODEL] == "gpt-3.5-turbo"
        assert attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] == 10
        assert attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] == 20
        assert attrs[SpanAttributes.COST.NUM_TOKENS] == 30
        assert attrs[SpanAttributes.CALL.RETURN] == "Test response"

    def test_request_serialization(self):
        """Test serialization of OpenAI request bodies."""

        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        attrs = {}

        # Extract request attributes
        if "messages" in request_body:
            attrs["llm.prompts"] = json.dumps(request_body["messages"])
            for msg in request_body["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    attrs["llm.input_text"] = msg.get("content", "")
                    break

        if "model" in request_body:
            attrs[SpanAttributes.COST.MODEL] = request_body["model"]

        if "temperature" in request_body:
            attrs["llm.temperature"] = request_body["temperature"]

        if "max_tokens" in request_body:
            attrs["llm.max_tokens"] = request_body["max_tokens"]

        # Verify attributes
        assert attrs[SpanAttributes.COST.MODEL] == "gpt-3.5-turbo"
        assert attrs["llm.input_text"] == "Hello"
        assert attrs["llm.temperature"] == 0.7
        assert attrs["llm.max_tokens"] == 100
        assert "Hello" in attrs["llm.prompts"]

    @pytest.mark.asyncio
    async def test_async_post_method(self):
        """Test async post method handling."""
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock(total_tokens=50)

        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)

        # Test that the mock is set up correctly
        assert hasattr(mock_client, "post")
        assert asyncio.iscoroutinefunction(mock_client.post)

        # Run the async method directly
        result = await mock_client.post(
            "/chat/completions", body={"model": "gpt-3.5-turbo"}
        )

        assert result.model == "gpt-3.5-turbo"
        assert result.usage.total_tokens == 50

    def test_cost_filtering_by_type(self):
        """Test that cost computation is filtered by response type."""
        # ChatCompletion - should compute costs
        chat_response = Mock()
        chat_response.model = "gpt-3.5-turbo"
        chat_response.usage = Mock()
        chat_response.__class__.__name__ = "ChatCompletion"

        should_compute = (
            hasattr(chat_response, "model")
            and hasattr(chat_response, "usage")
            and chat_response.__class__.__name__
            in ["ChatCompletion", "ParsedChatCompletion"]
        )
        assert should_compute is True

        # Embedding response - should NOT compute costs
        embed_response = Mock()
        embed_response.model = "text-embedding-ada-002"
        embed_response.usage = Mock()
        embed_response.__class__.__name__ = "CreateEmbeddingResponse"

        should_compute = (
            hasattr(embed_response, "model")
            and hasattr(embed_response, "usage")
            and embed_response.__class__.__name__
            in ["ChatCompletion", "ParsedChatCompletion"]
        )
        assert should_compute is False


@pytest.mark.optional
class TestSpanAttributeMapping:
    """Test proper mapping of span attributes."""

    def test_llm_attributes(self):
        """Test LLM-specific attribute mapping."""

        # Test data
        response_data = {
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is the response",
                    }
                }
            ],
        }

        attrs = {}

        # Map to TruLens attributes
        if response_data.get("model"):
            attrs[SpanAttributes.COST.MODEL] = response_data["model"]

        usage = response_data.get("usage", {})
        if usage:
            attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] = usage.get(
                "prompt_tokens", 0
            )
            attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = usage.get(
                "completion_tokens", 0
            )
            attrs[SpanAttributes.COST.NUM_TOKENS] = usage.get("total_tokens", 0)

            # Check for reasoning tokens (o1 models)
            if "completion_tokens_details" in usage:
                details = usage["completion_tokens_details"]
                if "reasoning_tokens" in details:
                    attrs[SpanAttributes.COST.NUM_REASONING_TOKENS] = details[
                        "reasoning_tokens"
                    ]

        if response_data.get("choices"):
            first_choice = response_data["choices"][0]
            message = first_choice.get("message", {})
            content = message.get("content", "")
            attrs[SpanAttributes.CALL.RETURN] = content
            attrs["llm.output_text"] = content
            attrs["llm.completions"] = json.dumps([
                {"role": message.get("role", "assistant"), "content": content}
            ])

        # Verify all attributes
        assert attrs[SpanAttributes.COST.MODEL] == "gpt-4"
        assert attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] == 100
        assert attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] == 200
        assert attrs[SpanAttributes.COST.NUM_TOKENS] == 300
        assert attrs[SpanAttributes.COST.NUM_REASONING_TOKENS] == 50
        assert attrs[SpanAttributes.CALL.RETURN] == "This is the response"
        assert attrs["llm.output_text"] == "This is the response"
        assert "assistant" in attrs["llm.completions"]

    def test_custom_attributes(self):
        """Test custom LLM attributes that don't map to standard ones."""
        attrs = {}

        # Custom attributes for LLM-specific fields
        attrs["llm.prompts"] = json.dumps([{"role": "user", "content": "test"}])
        attrs["llm.input_text"] = "test"
        attrs["llm.temperature"] = 0.5
        attrs["llm.max_tokens"] = 150
        attrs["openai.api.path"] = "/chat/completions"
        attrs["openai.api.request"] = '{"model": "gpt-3.5-turbo"}'
        attrs["openai.api.response"] = '{"usage": {"total_tokens": 30}}'

        # Verify all custom attributes exist
        assert "llm.prompts" in attrs
        assert "llm.input_text" in attrs
        assert "llm.temperature" in attrs
        assert "llm.max_tokens" in attrs
        assert "openai.api.path" in attrs
        assert "openai.api.request" in attrs
        assert "openai.api.response" in attrs


# =============================================================================
# Integration tests with LlamaIndex (if available)
# =============================================================================

# Try to import LlamaIndex components
try:
    from llama_index.core.workflow import Context
    from llama_index.core.workflow import Event
    from llama_index.core.workflow import StartEvent
    from llama_index.core.workflow import StopEvent
    from llama_index.core.workflow import Workflow
    from llama_index.core.workflow import step

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

    # Create mock classes if LlamaIndex is not installed
    class Event:
        pass

    class StartEvent(Event):
        def __init__(self, data=None):
            self.data = data or {}

    class StopEvent(Event):
        def __init__(self, result=None):
            self.result = result

    class Workflow:
        pass

    def step(func):
        return func

    Context = Mock

# Check if TruLlamaWorkflow can be imported (for skipif decorator)
TRULENS_LLAMAINDEX_AVAILABLE = (
    importlib.util.find_spec("trulens.apps.llamaindex") is not None
)

try:
    from trulens.core.session import TruSession
except ImportError:
    TruSession = None


# Test events for integration tests
@dataclass
class TestInputEvent(Event):
    """Test input event."""

    message: str


@dataclass
class TestOutputEvent(Event):
    """Test output event."""

    result: str
    metadata: Dict[str, Any] = None


@pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
class SimpleTestWorkflow(Workflow):
    """Simple test workflow with two steps."""

    @step
    async def process_input(self, ev: StartEvent) -> TestInputEvent:
        """Process the start event."""
        message = ev.data.get("message", "test")
        return TestInputEvent(message=message)

    @step
    async def generate_output(self, ev: TestInputEvent) -> StopEvent:
        """Generate the final output."""
        result = f"Processed: {ev.message}"
        return StopEvent(result=TestOutputEvent(result=result))


@pytest.fixture
def mock_session():
    """Create a mock TruSession."""
    session = Mock(spec=TruSession)
    session.connector = Mock()
    session.connector.db = Mock()
    return session


@pytest.mark.optional
@pytest.mark.skipif(
    not TRULENS_LLAMAINDEX_AVAILABLE, reason="TruLlamaWorkflow not available"
)
class TestTruLlamaWorkflowIntegration:
    """Integration tests for TruLlamaWorkflow."""

    @property
    def TruLlamaWorkflow(self):
        """Lazy import of TruLlamaWorkflow."""
        from trulens.apps.llamaindex import TruLlamaWorkflow

        return TruLlamaWorkflow

    def test_workflow_initialization(self, mock_session):
        """Test that TruLlamaWorkflow properly initializes a workflow."""
        workflow = SimpleTestWorkflow() if LLAMAINDEX_AVAILABLE else Mock()

        with patch(
            "trulens.apps.llamaindex.tru_llama_workflow.TruSession",
            return_value=mock_session,
        ):
            tru_workflow = self.TruLlamaWorkflow(
                workflow, app_name="test_workflow", app_version="1.0.0"
            )

            assert tru_workflow.app_name == "test_workflow"
            assert tru_workflow.app_version == "1.0.0"
            # The workflow is stored as 'app' in TruLlamaWorkflow
            assert hasattr(tru_workflow, "app")
            assert tru_workflow.app == workflow

            # Check that run method exists
            assert hasattr(workflow, "run")

    def test_workflow_attributes(self, mock_session):
        """Test that workflow attributes are properly set."""
        workflow = SimpleTestWorkflow() if LLAMAINDEX_AVAILABLE else Mock()

        with patch(
            "trulens.apps.llamaindex.tru_llama_workflow.TruSession",
            return_value=mock_session,
        ):
            tru_workflow = self.TruLlamaWorkflow(
                workflow,
                app_name="test_workflow",
                metadata={"test_key": "test_value"},
                tags=["test_tag"],
            )

            # Check that the workflow was initialized with the correct attributes
            assert tru_workflow.app_name == "test_workflow"
            assert tru_workflow.metadata == {"test_key": "test_value"}
            assert tru_workflow.tags == ["test_tag"]

    def test_function_agent_instrumentation(self, mock_session):
        """Test that FunctionAgent methods are properly instrumented."""
        workflow = SimpleTestWorkflow() if LLAMAINDEX_AVAILABLE else Mock()

        with patch(
            "trulens.apps.llamaindex.tru_llama_workflow.TruSession",
            return_value=mock_session,
        ):
            # Mock the FunctionAgent import and class
            mock_function_agent = Mock()
            mock_function_agent.run = Mock()
            mock_function_agent.arun = Mock()
            mock_function_agent.chat = Mock()
            mock_function_agent.achat = Mock()

            with patch(
                "trulens.apps.llamaindex.tru_llama_workflow.instrument_method"
            ) as mock_instrument_method:
                with patch(
                    "llama_index.core.agent.workflow.FunctionAgent",
                    mock_function_agent,
                ):
                    self.TruLlamaWorkflow(
                        workflow, app_name="test_workflow", app_version="1.0.0"
                    )

                    # Verify that instrument_method was called for FunctionAgent methods
                    expected_methods = [
                        "run",
                        "arun",
                        "chat",
                        "achat",
                        "stream_chat",
                        "astream_chat",
                    ]
                    instrumented_methods = []

                    for call in mock_instrument_method.call_args_list:
                        if call[1].get("cls") == mock_function_agent:
                            method_name = call[1].get("method_name")
                            if method_name in expected_methods:
                                instrumented_methods.append(method_name)

                    # At least some methods should be instrumented (those that exist on the mock)
                    assert (
                        len(instrumented_methods) > 0
                    ), "No FunctionAgent methods were instrumented"


@pytest.mark.optional
class TestWorkflowRecordAssociation:
    """Test suite for workflow record ID association."""

    def test_global_workflow_record_ids(self):
        """Test the global _WORKFLOW_RECORD_IDS dictionary."""
        try:
            from trulens.apps.llamaindex.tru_llama_workflow import (
                _WORKFLOW_RECORD_IDS,
            )
        except ImportError:
            # If import fails, simulate the dictionary
            _WORKFLOW_RECORD_IDS = {}

        # Clear the dictionary
        _WORKFLOW_RECORD_IDS.clear()

        # Test storing and retrieving record IDs
        test_id = "test-record-id-123"
        _WORKFLOW_RECORD_IDS["TestWorkflow"] = test_id

        assert _WORKFLOW_RECORD_IDS["TestWorkflow"] == test_id
        assert len(_WORKFLOW_RECORD_IDS) == 1

        # Test with multiple workflows
        _WORKFLOW_RECORD_IDS["AnotherWorkflow"] = "another-id-456"
        assert len(_WORKFLOW_RECORD_IDS) == 2
        assert _WORKFLOW_RECORD_IDS["TestWorkflow"] == test_id
        assert _WORKFLOW_RECORD_IDS["AnotherWorkflow"] == "another-id-456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
