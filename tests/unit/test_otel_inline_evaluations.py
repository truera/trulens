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
        # Create feedback function (higher is better by default).
        def simple_feedback(text: str) -> float:
            if text == "Kojikun":
                return 0.42
            return 0.0

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

        # Validate messages contain original content, guidance, and labeled result.
        def content_of(msg):
            return msg if isinstance(msg, str) else getattr(msg, "content", msg)

        assert content_of(result["messages"][0]) == "Nolan"
        assert content_of(result["messages"][1]) == "Sachiboy"

        guidance_msgs = [
            m
            for m in result["messages"]
            if isinstance(getattr(m, "content", None), str)
            and m.content.startswith("[Inline Evaluation Guidance]")
        ]
        assert len(guidance_msgs) == 1
        assert "higher is better" in guidance_msgs[0].content

        result_msgs = [
            m
            for m in result["messages"]
            if isinstance(getattr(m, "content", None), str)
            and m.content.startswith("[Inline Evaluation Result]")
        ]
        assert len(result_msgs) == 1
        assert "0.42" in result_msgs[0].content

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

    def test_guidance_direction_respects_higher_is_better_flag(self) -> None:
        # lower-is-better feedback
        def simple_feedback(text: str) -> float:
            return 0.25

        feedback_func = Feedback(simple_feedback, higher_is_better=False).on({
            "text": Selector(span_attribute="test_output")
        })

        @inline_evaluation(feedback_func, emit_spans=False)
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                "test_output": "anything"
            }
        )
        def simple_node(state: MessagesState) -> MessagesState:
            state["messages"].append(AIMessage("msg"))
            return state

        workflow = StateGraph(MessagesState)
        workflow.add_node("simple_node", simple_node)
        workflow.add_edge(START, "simple_node")
        workflow.add_edge("simple_node", END)
        graph = workflow.compile()

        tru_app = TruApp(graph, app_name="test_app", app_version="v1")
        initial_state = MessagesState(messages=["start"])
        with tru_app:
            result = graph.invoke(initial_state)

        guidance_msgs = [
            m
            for m in result["messages"]
            if isinstance(getattr(m, "content", None), str)
            and m.content.startswith("[Inline Evaluation Guidance]")
        ]
        assert len(guidance_msgs) == 1
        assert "lower is better" in guidance_msgs[0].content

    def test_guidance_added_only_once_across_multiple_nodes(self) -> None:
        # Feedback function
        def simple_feedback(text: str) -> float:
            return 0.5

        feedback_func = Feedback(simple_feedback).on({
            "text": Selector(span_attribute="test_output")
        })

        @inline_evaluation(feedback_func, emit_spans=False)
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                "test_output": "a"
            }
        )
        def node_a(state: MessagesState) -> MessagesState:
            state["messages"].append(AIMessage("A"))
            return state

        @inline_evaluation(feedback_func, emit_spans=False)
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                "test_output": "b"
            }
        )
        def node_b(state: MessagesState) -> MessagesState:
            state["messages"].append(AIMessage("B"))
            return state

        workflow = StateGraph(MessagesState)
        workflow.add_node("node_a", node_a)
        workflow.add_node("node_b", node_b)
        workflow.add_edge(START, "node_a")
        workflow.add_edge("node_a", "node_b")
        workflow.add_edge("node_b", END)
        graph = workflow.compile()

        tru_app = TruApp(graph, app_name="test_app", app_version="v1")
        initial_state = MessagesState(messages=["start"])
        with tru_app:
            result = graph.invoke(initial_state)

        guidance_msgs = [
            m
            for m in result["messages"]
            if isinstance(getattr(m, "content", None), str)
            and m.content.startswith("[Inline Evaluation Guidance]")
        ]
        assert len(guidance_msgs) == 1
