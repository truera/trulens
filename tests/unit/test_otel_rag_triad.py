from typing import List

import numpy as np
import pytest
from trulens.apps.app import TruApp
from trulens.core.feedback import Feedback
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _RAGApp:
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_context_helper(self, query: str) -> List[str]:
        return [
            "Babies are cute.",
            "Kojikun is widely considered the cutest baby in the world.",
            "Kojikun has only gotten cuter as time progresses.",
            "What?",
        ]

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_contexts(self, query: str) -> List[str]:
        return self.retrieve_context_helper(query)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        return "Kojikun!"

    @instrument()
    def query(self, question: str) -> str:
        if question != "Who is the cutest baby in the world?":
            raise ValueError("Invalid question!")
        contexts = self.retrieve_contexts(question)
        answer = self.generate_answer(question, contexts)
        return answer


@pytest.mark.optional
class TestOtelRagTriad(OtelTestCase):
    def test_rag_triad(self) -> None:
        # Create mock feedback functions.
        mock_groundedness_results = [0.99]
        mock_answer_relevance_results = [0.98]
        mock_context_relevance_results = [0.25, 1, 0.75, 0]

        def mock_groundedness(source: str, statement: str) -> float:
            return mock_groundedness_results.pop()

        def mock_answer_relevance(prompt: str, response: str) -> float:
            return mock_answer_relevance_results.pop()

        def mock_context_relevance(question: str, context: str) -> float:
            return mock_context_relevance_results.pop(0)

        # Create Feedbacks.
        f_groundedness = (
            Feedback(mock_groundedness, name="Groundedness")
            .on_context(collect_list=True)
            .on_output()
        )
        f_answer_relevance = (
            Feedback(mock_answer_relevance, name="Answer Relevance")
            .on_input()
            .on_output()
        )
        f_context_relevance = (
            Feedback(mock_context_relevance, name="Context Relevance")
            .on_input()
            .on_context(collect_list=False)
            .aggregate(np.mean)
        )
        # Create app.
        app = _RAGApp()
        tru_app = TruApp(
            app,
            app_name="RAG Triad App",
            app_version="v1",
            feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance],
        )
        # Record and invoke.
        tru_app.stop_evaluator()
        with tru_app:
            app.query("Who is the cutest baby in the world?")
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        # Verify mocks are called as expected.
        self.assertEqual(mock_groundedness_results, [])
        self.assertEqual(mock_answer_relevance_results, [])
        self.assertEqual(mock_context_relevance_results, [])
        # Verify the feedback function results.
        events = self._get_events()
        eval_roots = [
            curr
            for _, curr in events.iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        evals = [
            curr
            for _, curr in events.iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL
        ]
        self.assertEqual(3, len(eval_roots))
        self.assertListEqual(
            ["Answer Relevance", "Context Relevance", "Groundedness"],
            sorted([
                curr["record_attributes"][SpanAttributes.EVAL_ROOT.METRIC_NAME]
                for curr in eval_roots
            ]),
        )
        self.assertListEqual(
            [0.5, 0.98, 0.99],
            sorted([
                curr["record_attributes"][SpanAttributes.EVAL_ROOT.SCORE]
                for curr in eval_roots
            ]),
        )
        self.assertEqual(6, len(evals))
        self.assertListEqual(
            [0, 0.25, 0.75, 0.98, 0.99, 1],
            sorted([
                curr["record_attributes"][SpanAttributes.EVAL.SCORE]
                for curr in evals
            ]),
        )
