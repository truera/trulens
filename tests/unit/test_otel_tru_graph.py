"""
Tests for OTEL TruGraph app.
"""

import gc
import weakref

import pytest
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case
from tests.utils import enable_otel_backwards_compatibility

try:
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph
    from trulens.apps.langgraph import TruGraph  # noqa: F401

    LANGGRAPH_AVAILABLE = True
except Exception as e:
    print(f"LangGraph imports failed: {e}")
    LANGGRAPH_AVAILABLE = False


@pytest.mark.optional
@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
class TestOtelTruGraph(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_simple_multi_agent():
        """Helper function to create a simple multi-agent LangGraph."""

        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.agent_type": "research_agent"
            }
        )
        def research_agent(state):
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    query = last_message.content
                else:
                    query = str(last_message)

                response = f"Research findings for: {query}"
                return {"messages": [AIMessage(content=response)]}
            return {"messages": [AIMessage(content="No query provided")]}

        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.agent_type": "writer_agent"
            }
        )
        def writer_agent(state):
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    research_content = last_message.content
                else:
                    research_content = str(last_message)

                response = f"Article based on: {research_content}"
                return {"messages": [AIMessage(content=response)]}
            return {"messages": [AIMessage(content="No research provided")]}

        # Create workflow
        workflow = StateGraph(MessagesState)
        workflow.add_node("researcher", research_agent)
        workflow.add_node("writer", writer_agent)
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", END)
        workflow.set_entry_point("researcher")

        return workflow.compile()

    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = TestOtelTruGraph._create_simple_multi_agent()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.invoke, TruAppClass=TruGraph
        )

    def test_smoke(self) -> None:
        """Test basic TruGraph functionality."""

        multi_agent_graph = self._create_simple_multi_agent()
        tru_recorder = TruGraph(
            multi_agent_graph,
            app_name="Simple Multi-Agent",
            app_version="v1",
            main_method=multi_agent_graph.invoke,
        )

        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=(
                {"messages": [HumanMessage(content="What is AI?")]},
            ),
        )

        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_graph__test_smoke.csv"
        )

        # Check gc
        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del multi_agent_graph
        gc.collect()
        self.assertCollected(tru_recorder_ref)

    @enable_otel_backwards_compatibility
    def test_legacy_app(self) -> None:
        """Test TruGraph with legacy app interface."""

        multi_agent_graph = self._create_simple_multi_agent()
        tru_recorder = TruGraph(
            multi_agent_graph, app_name="Simple Multi-Agent", app_version="v1"
        )

        with tru_recorder:
            multi_agent_graph.invoke({
                "messages": [HumanMessage(content="What is AI?")]
            })

        # Compare results to expected.
        self._compare_record_attributes_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_graph__test_smoke.csv"
        )

    def test_auto_compilation(self) -> None:
        """Test that TruGraph automatically compiles uncompiled StateGraphs."""
        workflow = StateGraph(MessagesState)

        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello from agent")]}

        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        # TruGraph should automatically compile it
        tru_recorder = TruGraph(
            workflow,
            app_name="Auto Compiled",
            app_version="v1",
        )

        result = tru_recorder.app.invoke({
            "messages": [HumanMessage(content="Hello")]
        })
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_input_output_handling(self) -> None:
        """Test TruGraph input/output handling for different formats."""
        multi_agent_graph = self._create_simple_multi_agent()
        tru_recorder = TruGraph(
            multi_agent_graph,
            app_name="IO Test",
            app_version="v1",
        )

        result1 = tru_recorder.app.invoke({
            "messages": [HumanMessage(content="Test 1")]
        })
        assert "messages" in result1

        result2 = tru_recorder.app.invoke({"messages": [("user", "Test 2")]})
        assert "messages" in result2

        result3 = tru_recorder.main_call("Test 3")
        assert isinstance(result3, str)
        assert "Test 3" in result3

    def test_error_handling(self) -> None:
        """Test error handling when LangGraph is not available."""
        try:
            from trulens.apps.langgraph.tru_graph import LANGGRAPH_AVAILABLE

            assert (
                LANGGRAPH_AVAILABLE
            ), "LangGraph should be available in test environment"
        except ImportError:
            # This is expected if LangGraph is not installed
            pass
