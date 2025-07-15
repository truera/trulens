from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
import pytest
from trulens.apps.langgraph.inline_evaluations import inline_evaluation
from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import Selector

from tests.util.otel_test_case import OtelTestCase


@pytest.mark.optional
class TestOtelInlineEvaluations(OtelTestCase):
    def test_smoke(self) -> None:
        # Create feedback function.
        def simple_feedback(text: str) -> float:
            return 0.42

        feedback_func = Feedback(simple_feedback).on({
            "text": Selector(span_attribute="test_output")
        })

        # Create a simple langgraph app.
        @inline_evaluation(feedback_func)
        def simple_node(state: MessagesState) -> MessagesState:
            state.add_messages("Nolan")
            state.messages.append("Kojikun")
            return state

        workflow = StateGraph(MessagesState)
        workflow.add_node("simple_node", simple_node)
        workflow.add_edge(START, "simple_node")
        workflow.add_edge("simple_node", END)
        graph = workflow.compile()

        # Call the app.
        test_input = "Sachiboy"
        initial_state = MessagesState(messages=[test_input])
        result = graph.invoke(initial_state)

        # Ensure that when calling the app, the state is updated with the
        # feedback result
        self.assertListEqual(
            ["Sachiboy", "Kojikun", "Nolan", 0.42], result.messages
        )
