import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

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
    def _create_and_invoke_simple_app(self, emit_spans: bool) -> pd.DataFrame:
        # Create feedback function.
        def simple_feedback(text: str) -> float:
            if text == "Kojikun":
                return 0.42
            return 0

        feedback_func = Feedback(simple_feedback).on({
            "text": Selector(span_attribute="test_output")
        })

        # Create a simple langgraph app.
        @inline_evaluation(feedback_func, emit_spans=emit_spans)
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
        TruSession().force_flush()

        # Ensure that when calling the app, the state is updated with the
        # feedback result.
        self.assertListEqual(
            ["Nolan", "Sachiboy", "0.42"],
            [curr.content for curr in result["messages"]],
        )

        return self._get_events()

    def test_unemitted_spans(self) -> None:
        events = self._create_and_invoke_simple_app(emit_spans=False)
        self.assertListEqual(
            [SpanAttributes.SpanType.RECORD_ROOT],
            [
                curr[SpanAttributes.SPAN_TYPE]
                for curr in events["record_attributes"]
            ],
        )

    def test_emitted_spans(self) -> None:
        events = self._create_and_invoke_simple_app(emit_spans=True)
        self.assertListEqual(
            [
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.EVAL_ROOT,
                SpanAttributes.SpanType.EVAL,
            ],
            [
                curr[SpanAttributes.SPAN_TYPE]
                for curr in events["record_attributes"]
            ],
        )
        self.assertEqual(
            0.42,
            events.iloc[1]["record_attributes"][SpanAttributes.EVAL_ROOT.SCORE],
        )
