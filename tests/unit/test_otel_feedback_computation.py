"""
Tests for OTEL Feedback Computation.
"""

from typing import List

import pandas as pd
import pytest
from trulens.core.session import TruSession
from trulens.feedback.computer import MinimalSpanInfo
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback

from tests.util.mock_otel_feedback_computation import (
    all_retrieval_span_attributes,
)
from tests.util.mock_otel_feedback_computation import feedback_function
from tests.util.otel_test_case import OtelTestCase

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
class TestOtelFeedbackComputation(OtelTestCase):
    def test_feedback_computation(self) -> None:
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
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            ground_truth_output="Like attention but with more heads.",
            main_method_args=("What is multi-headed attention?",),
        )
        TruSession().force_flush()
        # Compute feedback on record we just ingested.
        events = self._get_events()
        spans = _convert_events_to_MinimalSpanInfos(events)
        record_root = RecordGraphNode.build_graph(spans)
        _compute_feedback(
            record_root,
            feedback_function,
            "baby_grader",
            all_retrieval_span_attributes,
        )
        TruSession().force_flush()
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_feedback_computation__test_feedback_computation.csv"
        )
