"""
Tests for LangGraph app detection and integration.
"""

import pytest
from trulens.core.session import TruSession

try:
    # These imports require optional dependencies to be installed.
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from trulens.apps.langgraph import TruGraph

    from langgraph.graph import END
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


@pytest.mark.optional
@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
class TestLangGraphDetection:
    """Test LangGraph automatic detection and integration."""

    def test_langgraph_detection_by_module(self):
        """Test that LangGraph apps are detected by module name."""

        # Create a simple LangGraph app
        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello from agent")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        # Test the module-based detection
        assert graph.__module__.startswith(
            "langgraph"
        ), f"Expected module to start with 'langgraph', got {graph.__module__}"

    def test_langgraph_detection_by_type(self):
        """Test that LangGraph apps are detected by type checking."""

        # Create a simple LangGraph app
        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello from agent")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        # Test the type-based detection
        session = TruSession()
        is_detected = session._is_langgraph_app(graph)
        assert is_detected, "LangGraph app should be detected by type checking"

    def test_session_auto_detection(self):
        """Test that TruSession automatically detects and creates TruGraph."""

        # Create a simple LangGraph app
        def echo_agent(state):
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    content = last_message.content
                else:
                    content = str(last_message)
                return {"messages": [AIMessage(content=f"Echo: {content}")]}
            return {"messages": [AIMessage(content="Echo: No message")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("echo", echo_agent)
        workflow.add_edge("echo", END)
        workflow.set_entry_point("echo")

        graph = workflow.compile()

        # Test automatic detection
        session = TruSession()
        tru_app = session.App(graph, app_name="DetectionTest")

        # Should return TruGraph instance
        assert isinstance(
            tru_app, TruGraph
        ), f"Expected TruGraph, got {type(tru_app)}"
        assert tru_app.app_name == "DetectionTest"

        # Test that it actually works
        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Hello")]
        })
        assert "messages" in result
        assert len(result["messages"]) > 0
        assert "Echo: Hello" in result["messages"][-1].content

    def test_manual_trugraph_creation(self):
        """Test manual TruGraph creation as fallback."""

        # Create a simple LangGraph app
        def simple_agent(state):
            return {"messages": [AIMessage(content="Manual creation works")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        # Test manual TruGraph creation
        tru_app = TruGraph(app=graph, app_name="ManualTest", app_version="1.0")

        assert isinstance(tru_app, TruGraph)
        assert tru_app.app_name == "ManualTest"
        assert tru_app.app_version == "1.0"

        # Test functionality
        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Test")]
        })
        assert "messages" in result
        assert "Manual creation works" in result["messages"][-1].content

    def test_uncompiled_graph_handling(self):
        """Test that TruGraph handles uncompiled StateGraphs."""
        # Create uncompiled workflow
        workflow = StateGraph(MessagesState)

        def simple_agent(state):
            return {"messages": [AIMessage(content="Auto-compiled")]}

        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        # Don't compile - let TruGraph handle it
        tru_app = TruGraph(
            app=workflow, app_name="UncompiledTest", app_version="1.0"
        )

        # Should work (TruGraph auto-compiles)
        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Test")]
        })
        assert "messages" in result
        assert "Auto-compiled" in result["messages"][-1].content

    def test_detection_attributes(self):
        """Test detection using attribute checking."""

        # Create a simple LangGraph app
        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        # Test attribute-based detection
        assert hasattr(
            graph, "graph"
        ), "Compiled graph should have 'graph' attribute"
        assert hasattr(
            graph, "invoke"
        ), "Compiled graph should have 'invoke' method"
        # Note: 'run' method might not exist in all LangGraph versions, so we check invoke instead

    def test_detection_with_different_graph_types(self):
        """Test detection with different LangGraph types."""

        # Test with compiled graph
        def agent1(state):
            return {"messages": [AIMessage(content="Agent 1")]}

        workflow1 = StateGraph(MessagesState)
        workflow1.add_node("agent", agent1)
        workflow1.add_edge("agent", END)
        workflow1.set_entry_point("agent")

        compiled_graph = workflow1.compile()

        # Test with uncompiled graph
        def agent2(state):
            return {"messages": [AIMessage(content="Agent 2")]}

        workflow2 = StateGraph(MessagesState)
        workflow2.add_node("agent", agent2)
        workflow2.add_edge("agent", END)
        workflow2.set_entry_point("agent")

        uncompiled_graph = workflow2

        # Both should be detected
        session = TruSession()

        is_compiled_detected = session._is_langgraph_app(compiled_graph)
        is_uncompiled_detected = session._is_langgraph_app(uncompiled_graph)

        assert is_compiled_detected, "Compiled graph should be detected"
        assert is_uncompiled_detected, "Uncompiled graph should be detected"

        # Both should create TruGraph instances
        tru_app1 = session.App(compiled_graph, app_name="Compiled")
        tru_app2 = session.App(uncompiled_graph, app_name="Uncompiled")

        assert isinstance(tru_app1, TruGraph)
        assert isinstance(tru_app2, TruGraph)

        # Both should work
        result1 = tru_app1.app.invoke({
            "messages": [HumanMessage(content="Test1")]
        })
        result2 = tru_app2.app.invoke({
            "messages": [HumanMessage(content="Test2")]
        })

        assert "messages" in result1
        assert "messages" in result2
        assert "Agent 1" in result1["messages"][-1].content
        assert "Agent 2" in result2["messages"][-1].content
