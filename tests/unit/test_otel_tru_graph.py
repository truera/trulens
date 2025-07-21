"""
Tests for OTEL TruGraph app.
"""

import gc
import time
import uuid
import weakref

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.func import task
from langgraph.graph import END
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
import pytest
from trulens.apps.langgraph import TruGraph
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case


@pytest.mark.optional
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
                return {
                    "messages": [
                        AIMessage(content=response, id=2),
                    ]
                }
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
                return {
                    "messages": [
                        AIMessage(content=response, id="3"),
                    ]
                }
            return {"messages": [AIMessage(content="No research provided")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("researcher", research_agent)
        workflow.add_node("writer", writer_agent)
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", END)
        workflow.set_entry_point("researcher")

        return workflow.compile()

    @staticmethod
    def _create_functional_api_graph_app():
        """Helper function to create a LangGraph app using Function API."""

        @task
        def write_essay(topic: str) -> str:
            time.sleep(0.1)  # Simulate work
            return f"An essay about {topic}"

        @entrypoint(checkpointer=MemorySaver())
        def workflow(topic: str) -> dict:
            essay = write_essay(topic).result()

            return {
                "essay": essay,
                "is_approved": True,
            }

        class EssayWriter:
            def __init__(self):
                self.workflow = workflow

            def run(self, topic: str) -> dict:
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                return self.workflow.invoke(topic, config)

        return EssayWriter()

    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = TestOtelTruGraph._create_simple_multi_agent()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.invoke, TruAppClass=TruGraph
        )

    def test_smoke(self) -> None:
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
                {"messages": [HumanMessage(content="What is AI?", id="1")]},
            ),
        )

        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_graph_test_smoke.csv"
        )

        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del multi_agent_graph
        gc.collect()
        self.assertCollected(tru_recorder_ref)

    def test_task_instrumentation(self) -> None:
        essay_writer = self._create_functional_api_graph_app()

        tru_recorder = TruGraph(
            essay_writer,
            main_method=essay_writer.run,
            app_name="Essay Writer",
            app_version="v1",
        )

        result = tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("artificial intelligence",),
        )

        self.assertIn("essay", result)
        self.assertIn("is_approved", result)
        self.assertIn("artificial intelligence", result["essay"])

        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_graph_test_function_api_smoke.csv"
        )

    def test_langgraph_detection_by_module(self):
        """Test that LangGraph apps are detected by module name."""

        def simple_agent(state):
            return {"messages": [AIMessage(content="Hello from agent")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        graph = workflow.compile()

        self.assertIn("langgraph", graph.__module__)

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

        session = TruSession()
        tru_app = session.App(graph, app_name="DetectionTest")

        self.assertIsInstance(tru_app, TruGraph)
        self.assertEqual(tru_app.app_name, "DetectionTest")

        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Hello")]
        })
        self.assertIn("messages", result)
        self.assertGreater(len(result["messages"]), 0)
        self.assertIn("Echo: Hello", result["messages"][-1].content)

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

        self.assertIsInstance(tru_app, TruGraph)
        self.assertEqual(tru_app.app_name, "ManualTest")
        self.assertEqual(tru_app.app_version, "1.0")

        result = tru_app.app.invoke({
            "messages": [HumanMessage(content="Test")]
        })
        self.assertIn("messages", result)
        self.assertIn("Manual creation works", result["messages"][-1].content)

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
        self.assertIn("messages", result1)

        result2 = tru_recorder.app.invoke({"messages": [("user", "Test 2")]})
        self.assertIn("messages", result2)

        result3 = tru_recorder.main_call("Test 3")
        self.assertIsInstance(result3, str)
        self.assertIn("Test 3", result3)

    def test_custom_class_support(self) -> None:
        """Test TruGraph support for custom classes with LangGraph workflows."""

        def simple_agent(state):
            return {"messages": [AIMessage(content="Custom class response")]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", simple_agent)
        workflow.add_edge("agent", END)
        workflow.set_entry_point("agent")

        class CustomAgent:
            def __init__(self):
                self.workflow = workflow.compile()

            def run(self, input_data):
                return self.workflow.invoke(input_data)

        custom_agent = CustomAgent()
        tru_recorder = TruGraph(
            custom_agent,
            app_name="Custom Agent",
            app_version="v1",
        )

        result = tru_recorder.app.run({"messages": [("user", "Test input")]})
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("Custom class response", result["messages"][-1].content)
