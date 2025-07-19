"""
Tests for LangGraph app detection and integration.
"""

import pytest
from trulens.core.session import TruSession

try:
    # These imports require optional dependencies to be installed.
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
class TestLangGraphDetection:
    """Test LangGraph automatic detection and integration."""

    def test_langgraph_detection_by_module(self):
        """Test that LangGraph apps are detected by module name."""

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

    def test_session_auto_detection(self):
        """Test that TruSession automatically detects and creates TruGraph."""

        def echo_agent(state):
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
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

        assert isinstance(
            tru_app, TruGraph
        ), f"Expected TruGraph, got {type(tru_app)}"

        assert tru_app.app_name == "DetectionTest"

        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Hello")]
        })
        assert "messages" in result
        assert len(result["messages"]) > 0
        assert "Echo: Hello" in result["messages"][-1].content

    def test_manual_trugraph_creation(self):
        """Test manual TruGraph creation as fallback."""

        def simple_agent(state):
            return {"messages": [AIMessage(content="Manual creation works")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

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

        # let TruGraph handle it without explicitly compiling
        tru_app = TruGraph(
            app=workflow, app_name="UncompiledTest", app_version="1.0"
        )

        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Test")]
        })
        assert "messages" in result
        assert "Auto-compiled" in result["messages"][-1].content

    def test_detection_attributes(self):
        """Test detection using attribute checking."""

        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        # Test attribute-based detection
        assert hasattr(
            graph, "invoke"
        ), "Compiled graph should have 'invoke' method"
