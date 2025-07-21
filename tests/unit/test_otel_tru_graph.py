"""
Tests for OTEL TruGraph app.
"""

import gc
import weakref

import pytest
from trulens.core.session import TruSession

import tests.util.otel_tru_app_test_case

try:
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langgraph.func import entrypoint
    from langgraph.func import task
    from langgraph.graph import END
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph
    from trulens.apps.langgraph import TruGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    pass


@pytest.mark.optional
class TestOtelTruGraph(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_simple_multi_agent():
        """Helper function to create a simple multi-agent graph."""

        def research(state):
            query = state["messages"][-1].content if state["messages"] else None
            if query:
                response = f"Research findings for: {query}"
                return {
                    "messages": [
                        AIMessage(content=response, id=2),
                    ]
                }
            return {"messages": [AIMessage(content="No query provided")]}

        def write_article(state):
            research_content = (
                state["messages"][-1].content if state["messages"] else None
            )
            if research_content:
                response = f"Article based on: {research_content}"
                return {
                    "messages": [
                        AIMessage(content=response, id="3"),
                    ]
                }
            return {"messages": [AIMessage(content="No research provided")]}

        workflow = StateGraph(MessagesState)

        workflow.add_node("research", research)
        workflow.add_node("write", write_article)

        workflow.set_entry_point("research")
        workflow.add_edge("research", "write")
        workflow.add_edge("write", END)

        return workflow.compile()

    @staticmethod
    def _create_functional_api_graph_app():
        """Helper function to create a LangGraph app using Function API."""

        @task(name="research")
        def research(topic: str) -> str:
            return f"Research findings for: {topic}"

        @task(name="write_essay")
        def write_essay(research_findings: str) -> dict:
            essay = f"Essay based on: {research_findings}"
            return {"essay": essay, "is_approved": True}

        @entrypoint(name="workflow")
        def workflow(topic: str) -> dict:
            findings = research(topic)
            result = write_essay(findings)
            return result

        return workflow

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
        """Test that LangGraph is detected by module name."""
        graph = self._create_simple_multi_agent()

        self.assertIn("langgraph", graph.__module__)

    def test_session_auto_detection(self):
        """Test that TruGraph is automatically detected and created."""
        session = TruSession()
        graph = self._create_simple_multi_agent()

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
        """Test manual creation of TruGraph."""
        graph = self._create_simple_multi_agent()

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
        """Test handling of different input/output formats."""
        tru_recorder = TruGraph(
            app=self._create_simple_multi_agent(),
            app_name="IOTest",
            app_version="1.0",
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
        """Test support for custom classes."""

        class CustomGraph:
            def run(self, input_data):
                return {
                    "messages": [AIMessage(content="Custom class response")]
                }

        tru_recorder = TruGraph(
            app=CustomGraph(),
            app_name="CustomTest",
            app_version="1.0",
        )

        result = tru_recorder.app.run({"messages": [("user", "Test input")]})
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("Custom class response", result["messages"][-1].content)

    def test_with_existing_tru_session(self):
        """Test using TruGraph with an existing TruSession."""
        session = TruSession()
        graph = self._create_simple_multi_agent()

        tru_recorder = TruGraph(
            app=graph,
            app_name="SessionTest",
            app_version="1.0",
            tru_session=session,
        )

        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        gc.collect()

        self.assertCollected(tru_recorder_ref)

    def test_with_new_tru_session(self):
        """Test using TruGraph with a new TruSession."""
        graph = self._create_simple_multi_agent()

        tru_recorder = TruGraph(
            app=graph,
            app_name="NewSessionTest",
            app_version="1.0",
        )

        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        gc.collect()

        self.assertCollected(tru_recorder_ref)
