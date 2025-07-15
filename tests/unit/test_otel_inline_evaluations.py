import pytest
from trulens.apps.app import TruApp
from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument

from tests.util.otel_test_case import OtelTestCase

try:
    from langchain_core.messages import AIMessage
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph
    from trulens.apps.langgraph.inline_evaluations import inline_evaluation
except Exception:
    pass


@pytest.mark.optional
class TestOtelInlineEvaluations(OtelTestCase):
    def test_smoke(self) -> None:
        # Create feedback function.
        def simple_feedback(text: str) -> float:
            if text == "Kojikun":
                return 0.42
            return 0

        feedback_func = Feedback(simple_feedback).on({
            "text": Selector(span_attribute="test_output")
        })

        # Create a simple langgraph app.
        @inline_evaluation(feedback_func)
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                "test_output": "Kojikun"
            }
        )
        def simple_node(state: MessagesState) -> MessagesState:
            state["messages"].append(AIMessage("Sachiboy"))
            return state

        workflow = StateGraph(MessagesState)
        workflow.add_node("simple_node", simple_node)
        workflow.add_edge(START, "simple_node")
        workflow.add_edge("simple_node", END)
        graph = workflow.compile()

        tru_app = TruApp(graph, app_name="test_app", app_version="v1")

        # Call the app.
        initial_state = MessagesState(messages=["Nolan"])
        with tru_app:
            result = graph.invoke(initial_state)

        # Ensure that when calling the app, the state is updated with the
        # feedback result
        self.assertListEqual(
            ["Nolan", "Sachiboy", "0.42"],
            [curr.content for curr in result["messages"]],
        )
