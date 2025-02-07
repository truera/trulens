"""
Tests for OTEL Feedback Computation.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.feedback.computer import MinimalSpanInfo
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase

try:
    # These imports require optional dependencies to be installed.
    from trulens.apps.langchain import TruChain

    import tests.unit.test_otel_tru_chain
except Exception:
    pass


def _convert_events_to_MinimalSpanInfos(
    events: pd.DataFrame,
) -> List[MinimalSpanInfo]:
    ret = []
    for _, row in events.iterrows():
        span = MinimalSpanInfo()
        span.span_id = row["trace"]["span_id"]
        span.parent_span_id = row["record"]["parent_span_id"]
        if not span.parent_span_id:
            span.parent_span_id = None
        span.attributes = row["record_attributes"]
        ret.append(span)
    return ret


@pytest.mark.optional
class TestOtelFeedbackComputation(OtelAppTestCase):
    def test_feedback_computation(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app.
        rag_chain = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            rag_chain.invoke("What is multi-headed attention?")
        tru_session.force_flush()
        # Get events.
        events = self._get_events()
        # Convert events to list of `MinimalSpanInfo`.
        spans = _convert_events_to_MinimalSpanInfos(events)
        # Build record graph.
        record_root = RecordGraphNode.build_graph(spans)

        # Define selector function.
        def all_retrieval_span_attributes(
            node: RecordGraphNode,
        ) -> List[Dict[str, Any]]:
            ret = []
            if (
                node.current_span.attributes.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RETRIEVAL
            ):
                ret = [node.current_span.attributes]
            # Recurse on children.
            for child in node.children_spans:
                ret.extend(all_retrieval_span_attributes(child))
            return ret

        # Define feedback function.
        @instrument(
            span_type=SpanAttributes.SpanType.EVAL_CHILD,
            full_scoped_attributes=lambda ret, exception, *args, **kwargs: {
                SpanAttributes.EVAL_CHILD.CRITERIA: kwargs["criteria"],
                SpanAttributes.EVAL_CHILD.EVIDENCE: ret[1],
                SpanAttributes.EVAL_CHILD.SCORE: ret[0],
            },
        )
        def child_feedback_function(criteria: str) -> Tuple[float, str]:
            return {
                "Is Kojikun the best baby?": (
                    1,
                    "Kojikun is extraordinarily cute and fun and is the best baby!",
                ),
                "Is Nolan the best baby?": (
                    0.42,
                    "Nolan is just another name for Kojikun so he is the best baby!",
                ),
            }[criteria]

        def feedback_function(**kwargs):
            child_feedback_function("Is Kojikun the best baby?")
            child_feedback_function("Is Nolan the best baby?")
            return 0.42, {"best_baby": "Kojikun/Nolan"}

        # Compute feedback.
        _compute_feedback(
            record_root, feedback_function, all_retrieval_span_attributes
        )
        tru_session.force_flush()
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_feedback_computation__test_feedback_computation.csv"
        )
