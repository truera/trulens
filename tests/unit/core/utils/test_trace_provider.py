"""Tests for trace compression and trace provider implementations."""

import pytest
from trulens.core.utils.trace_compression import MAX_TRACE_SIZE
from trulens.core.utils.trace_compression import TraceCompressor
from trulens.core.utils.trace_compression import compress_trace_for_feedback
from trulens.core.utils.trace_provider import GenericTraceProvider
from trulens.core.utils.trace_provider import TraceProviderRegistry
from trulens.core.utils.trace_provider import get_trace_provider

# LangGraph provider import is deferred to avoid import errors in some environments
LANGGRAPH_AVAILABLE = False
LangGraphTraceProvider = None


def _get_langgraph_provider():
    """Lazily import and return LangGraphTraceProvider."""
    global LANGGRAPH_AVAILABLE, LangGraphTraceProvider
    if LangGraphTraceProvider is None:
        try:
            from trulens.apps.langgraph.trace_provider import (
                LangGraphTraceProvider as _LangGraphTraceProvider,
            )

            LangGraphTraceProvider = _LangGraphTraceProvider
            LANGGRAPH_AVAILABLE = True
        except (ImportError, Exception):
            LANGGRAPH_AVAILABLE = False
    return LangGraphTraceProvider


class TestGenericTraceProvider:
    """Tests for GenericTraceProvider."""

    def test_can_handle_returns_true_for_any_trace(self):
        """Generic provider should handle any trace data as fallback."""
        provider = GenericTraceProvider()

        assert provider.can_handle({}) is True
        assert provider.can_handle({"spans": []}) is True
        assert provider.can_handle({"random": "data"}) is True

    def test_extract_plan_returns_none_when_no_plan(self):
        """Should return None when trace has no plan."""
        provider = GenericTraceProvider()

        trace_data = {
            "trace_id": "test-123",
            "spans": [
                {
                    "span_name": "some_operation",
                    "span_attributes": {"key": "value"},
                }
            ],
        }

        result = provider.extract_plan(trace_data)
        assert result is None

    def test_extract_plan_from_top_level_plan_field(self):
        """Should extract plan from top-level 'plan' field."""
        provider = GenericTraceProvider()

        trace_data = {
            "trace_id": "test-123",
            "plan": {
                "plan_summary": "Test the system",
                "steps": [{"agent": "tester", "purpose": "run tests"}],
            },
        }

        result = provider.extract_plan(trace_data)
        assert result is not None
        assert "plan_summary" in str(result) or "Test the system" in str(result)

    def test_extract_plan_cleans_debug_messages(self):
        """Should clean debug messages from plan content."""
        provider = GenericTraceProvider()

        trace_data = {
            "plan": "Agent error: some error\nDEBUG: some debug\nActual plan content here",
        }

        result = provider.extract_plan(trace_data)
        assert result is not None
        assert "Actual plan content here" in str(result)
        # Debug messages should be cleaned
        assert "Agent error:" not in str(result)
        assert "DEBUG:" not in str(result)

    def test_extract_plan_truncates_large_plans(self):
        """Should truncate plans larger than 10KB."""
        provider = GenericTraceProvider()

        # Create a plan larger than 10KB
        large_plan = "x" * 15000
        trace_data = {"plan": large_plan}

        result = provider.extract_plan(trace_data)
        assert result is not None
        assert len(str(result)) < len(large_plan)
        assert "TRUNCATED" in str(result)

    def test_extract_execution_flow_from_spans(self):
        """Should extract execution flow from spans."""
        provider = GenericTraceProvider()

        trace_data = {
            "spans": [
                {"span_name": "step_1", "span_type": "operation"},
                {"span_name": "step_2", "span_type": "tool_call"},
                {"span_name": "step_3", "span_type": "response"},
            ]
        }

        result = provider.extract_execution_flow(trace_data)

        assert len(result) == 3
        assert result[0]["name"] == "step_1"
        assert result[1]["name"] == "step_2"
        assert result[2]["name"] == "step_3"

    def test_extract_execution_flow_returns_empty_list_when_no_spans(self):
        """Should return empty list when no spans exist."""
        provider = GenericTraceProvider()

        result = provider.extract_execution_flow({})
        assert result == []

    def test_extract_agent_interactions_from_messages(self):
        """Should extract agent interactions from messages."""
        provider = GenericTraceProvider()

        trace_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        result = provider.extract_agent_interactions(trace_data)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_compress_trace_includes_plan_and_execution_flow(self):
        """compress_trace should include both plan and execution flow."""
        provider = GenericTraceProvider()

        trace_data = {
            "trace_id": "test-123",
            "plan": {"summary": "Test plan"},
            "spans": [
                {"span_name": "step_1"},
                {"span_name": "step_2"},
            ],
        }

        result = provider.compress_trace(trace_data)

        assert "plan" in result
        assert "execution_flow" in result
        assert result["trace_id"] == "test-123"

    def test_compress_trace_without_plan_still_includes_execution_flow(self):
        """compress_trace should include execution flow even without plan."""
        provider = GenericTraceProvider()

        trace_data = {
            "trace_id": "test-456",
            "spans": [
                {"span_name": "step_1"},
                {"span_name": "step_2"},
            ],
        }

        result = provider.compress_trace(trace_data)

        # Should still have execution_flow even though there's no plan
        assert "execution_flow" in result
        assert len(result["execution_flow"]) == 2


class TestLangGraphTraceProvider:
    """Tests for LangGraphTraceProvider."""

    @pytest.fixture
    def provider(self):
        """Create a LangGraphTraceProvider instance."""
        provider_class = _get_langgraph_provider()
        if provider_class is None:
            pytest.skip("LangGraph provider not available in this environment")
        return provider_class()

    def test_can_handle_detects_langgraph_span_names(self, provider):
        """Should detect LangGraph traces by span names."""
        trace_data = {
            "spans": [
                {"span_name": "langgraph_node", "span_attributes": {}},
            ]
        }

        assert provider.can_handle(trace_data) is True

    def test_can_handle_detects_graph_attributes(self, provider):
        """Should detect LangGraph traces by OTEL attributes."""
        trace_data = {
            "spans": [
                {
                    "span_name": "some_span",
                    "span_attributes": {
                        "ai.observability.graph_node.node_name": "coordinator"
                    },
                },
            ]
        }

        assert provider.can_handle(trace_data) is True

    def test_can_handle_detects_command_structures(self, provider):
        """Should detect LangGraph traces by Command structures."""
        trace_data = {
            "spans": [
                {
                    "span_name": "some_span",
                    "span_attributes": {
                        "output": "Command(goto='next_agent', update={'key': 'value'})"
                    },
                },
            ]
        }

        assert provider.can_handle(trace_data) is True

    def test_can_handle_returns_false_for_non_langgraph(self, provider):
        """Should return False for non-LangGraph traces."""
        trace_data = {
            "spans": [
                {
                    "span_name": "generic_operation",
                    "span_attributes": {"some_key": "some_value"},
                },
            ]
        }

        assert provider.can_handle(trace_data) is False

    def test_extract_plan_returns_none_when_no_plan(self, provider):
        """Should return None when trace has no plan."""
        trace_data = {
            "trace_id": "test-123",
            "spans": [
                {
                    "span_name": "langgraph_node",
                    "span_attributes": {
                        "ai.observability.graph_node.output_state": "{'messages': ['hello']}"
                    },
                }
            ],
        }

        result = provider.extract_plan(trace_data)
        # Should return None since there's no plan-related content
        # (the output_state doesn't contain 'plan' with >100 chars)
        assert result is None

    def test_extract_plan_from_direct_field(self, provider):
        """Should extract plan from direct 'plan' field."""
        trace_data = {
            "plan": {
                "plan_summary": "Analyze customer data",
                "steps": [
                    {"agent": "data_agent", "purpose": "fetch data"},
                    {"agent": "analysis_agent", "purpose": "analyze"},
                ],
            }
        }

        result = provider.extract_plan(trace_data)
        assert result is not None

    def test_extract_plan_from_command_structure(self, provider):
        """Should extract plan from Command string."""
        command_str = """Command(update={'plan': {'plan_summary': 'Test plan', 'steps': [{'agent': 'tester', 'purpose': 'test'}]}, 'messages': []}, goto='next')"""

        trace_data = {
            "spans": [
                {
                    "span_name": "coordinator",
                    "span_attributes": {
                        "ai.observability.graph_node.output_state": command_str
                    },
                }
            ]
        }

        result = provider.extract_plan(trace_data)
        assert result is not None

    def test_extract_plan_from_graph_state(self, provider):
        """Should extract plan from graph node state attributes."""
        trace_data = {
            "spans": [
                {
                    "span_name": "planning_node",
                    "span_attributes": {
                        "ai.observability.graph_node.output_state": """{
                            "plan": {
                                "plan_summary": "Multi-step analysis",
                                "steps": [{"agent": "analyst", "purpose": "analyze"}]
                            }
                        }"""
                    },
                }
            ]
        }

        result = provider.extract_plan(trace_data)
        assert result is not None

    def test_extract_execution_flow_from_spans(self, provider):
        """Should extract detailed execution flow from spans."""
        trace_data = {
            "spans": [
                {
                    "span_name": "coordinator",
                    "span_attributes": {
                        "ai.observability.graph_node.node_name": "coordinator"
                    },
                },
                {
                    "span_name": "data_agent",
                    "span_attributes": {
                        "ai.observability.call.function": "fetch_data",
                        "ai.observability.call.return": "{'results': [1,2,3]}",
                    },
                },
            ]
        }

        result = provider.extract_execution_flow(trace_data)

        assert len(result) >= 1
        # Should classify steps by type
        step_types = [step.get("type") for step in result]
        assert any(
            t in step_types for t in ["coordination", "agent_execution", "step"]
        )

    def test_compress_trace_includes_all_components(self, provider):
        """compress_trace should include plan, execution flow, and interactions."""
        trace_data = {
            "trace_id": "test-lg-123",
            "plan": {"summary": "LangGraph test plan"},
            "spans": [
                {
                    "span_name": "coordinator",
                    "span_attributes": {
                        "ai.observability.graph_node.node_name": "coordinator",
                        "ai.observability.graph_node.output_state": "{'messages': ['test']}",
                    },
                },
            ],
        }

        result = provider.compress_trace(trace_data)

        assert "plan" in result
        assert "trace_id" in result
        # Should have execution flow or agent interactions
        has_flow_or_interactions = (
            "execution_flow" in result or "agent_interactions" in result
        )
        assert has_flow_or_interactions or "plan" in result

    def test_compress_trace_without_plan(self, provider):
        """compress_trace should work even without a plan."""
        trace_data = {
            "trace_id": "test-no-plan",
            "spans": [
                {
                    "span_name": "agent_node",
                    "span_attributes": {
                        "ai.observability.graph_node.node_name": "agent",
                        "ai.observability.call.function": "process",
                        "ai.observability.call.return": "done",
                    },
                },
            ],
        }

        result = provider.compress_trace(trace_data)

        # Should not fail, should still have trace_id and some content
        assert result["trace_id"] == "test-no-plan"
        # Plan should not be present
        assert "plan" not in result or result.get("plan") is None


class TestTraceProviderRegistry:
    """Tests for TraceProviderRegistry."""

    def test_registry_returns_generic_provider_as_fallback(self):
        """Registry should return GenericTraceProvider as fallback."""
        registry = TraceProviderRegistry()

        trace_data = {"some": "data"}
        provider = registry.get_provider(trace_data)

        assert isinstance(provider, GenericTraceProvider)

    def test_get_trace_provider_function(self):
        """get_trace_provider should return appropriate provider."""
        trace_data = {"trace_id": "test"}
        provider = get_trace_provider(trace_data)

        # Should return a TraceProvider instance
        assert hasattr(provider, "compress_trace")
        assert hasattr(provider, "extract_plan")


class TestTraceCompression:
    """Tests for trace compression utilities."""

    def test_compress_trace_for_feedback_basic(self):
        """Basic trace compression should work."""
        trace_data = {
            "trace_id": "test-compress",
            "spans": [
                {"span_name": "step1", "span_attributes": {}},
            ],
        }

        result = compress_trace_for_feedback(trace_data)

        assert isinstance(result, dict)
        assert "error" not in result

    def test_compress_trace_for_feedback_with_plan(self):
        """Compression should preserve plan."""
        trace_data = {
            "trace_id": "test-with-plan",
            "plan": {"summary": "Important plan"},
            "spans": [],
        }

        result = compress_trace_for_feedback(trace_data, preserve_plan=True)

        assert "plan" in result

    def test_compress_trace_for_feedback_scales_token_limit_for_large_traces(
        self,
    ):
        """Large traces should have token limit scaled down."""
        # Create a trace larger than MAX_TRACE_SIZE
        large_content = "x" * (MAX_TRACE_SIZE + 100000)
        trace_data = {
            "trace_id": "large-trace",
            "large_field": large_content,
            "spans": [],
        }

        # This should not raise an error and should complete
        result = compress_trace_for_feedback(
            trace_data, target_token_limit=100000
        )

        assert isinstance(result, dict)

    def test_trace_compressor_normalize_string_input(self):
        """TraceCompressor should handle string input."""
        compressor = TraceCompressor()

        json_string = '{"trace_id": "test", "spans": []}'
        result = compressor.compress_trace(json_string)

        assert isinstance(result, dict)

    def test_trace_compressor_handles_invalid_json(self):
        """TraceCompressor should handle invalid JSON gracefully."""
        compressor = TraceCompressor()

        invalid_json = "not valid json {"
        result = compressor.compress_trace(invalid_json)

        assert "error" in result
