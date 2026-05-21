"""Integration test for the RAG triad full pipeline using a MockLLMProvider.

Verifies selector resolution, score aggregation, and EVAL/EVAL_ROOT span
creation end-to-end.  Complements tests/unit/test_otel_rag_triad.py, which
uses bare mock functions that bypass Metric/Selector wiring entirely.

Uses ``Metric`` (not the deprecated ``Feedback`` alias) per the current API.
"""

from typing import ClassVar

import numpy as np
import pytest
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.feedback import llm_provider
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _RAGApp:
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_context_helper(self, query: str) -> list[str]:
        return [
            "Babies are cute.",
            "Kojikun is widely considered the cutest baby in the world.",
            "Kojikun has only gotten cuter as time progresses.",
            "What?",
        ]

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve_contexts(self, query: str) -> list[str]:
        return self.retrieve_context_helper(query)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_answer(self, query: str, retrieved_docs: list[str]) -> str:
        return "Kojikun!"

    @instrument()
    def query(self, question: str) -> str:
        if question != "Who is the cutest baby in the world?":
            raise ValueError("Invalid question!")
        contexts = self.retrieve_contexts(question)
        answer = self.generate_answer(question, contexts)
        return answer


class MockLLMProvider(llm_provider.LLMProvider):
    """LLMProvider that returns deterministic scores without a live endpoint.

    Overrides the three top-level RAG triad methods directly so the endpoint
    assertion inside ``_remove_trivial_statements`` is never reached.  Call
    logs on each instance are used to assert selector resolution.
    """

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    CONTEXT_SCORES: ClassVar[dict[str, float]] = {
        "Babies are cute.": 0.25,
        "Kojikun is widely considered the cutest baby in the world.": 1.0,
        "Kojikun has only gotten cuter as time progresses.": 0.75,
        "What?": 0.0,
    }

    def __init__(self, **kwargs):
        super().__init__(
            endpoint=None,
            model_engine="mock-model",
            **kwargs,
        )
        # Per-instance call logs; used to verify selector resolution.
        self.groundedness_calls: list[tuple] = []
        self.relevance_calls: list[tuple] = []
        self.context_relevance_calls: list[tuple] = []

    def groundedness_measure_with_cot_reasons(
        self,
        source: str,
        statement: str,
        **kwargs,
    ) -> tuple[float, dict]:
        self.groundedness_calls.append((source, statement))
        return 0.8, {"reason": "Mock groundedness reason."}

    def relevance(
        self,
        prompt: str,
        response: str,
        **kwargs,
    ) -> float:
        self.relevance_calls.append((prompt, response))
        return 0.8

    def context_relevance(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> float:
        self.context_relevance_calls.append((question, context))
        return self.CONTEXT_SCORES.get(context, 0.5)


@pytest.mark.optional
class TestRagTriadPipeline(OtelTestCase):
    def test_rag_triad_full_pipeline(self) -> None:
        provider = MockLLMProvider()

        f_groundedness = Metric(
            implementation=provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        )
        f_answer_relevance = Metric(
            implementation=provider.relevance,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        )
        f_context_relevance = Metric(
            implementation=provider.context_relevance,
            name="Context Relevance",
            agg=np.mean,
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
        )

        app = _RAGApp()
        tru_app = TruApp(
            app,
            app_name="RAG Triad Pipeline",
            app_version="v1",
            feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
        )
        tru_app.stop_evaluator()
        with tru_app:
            app.query("Who is the cutest baby in the world?")
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        events = self._get_events()
        eval_roots = [
            curr
            for _, curr in events.iterrows()
            if curr["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        evals = [
            curr
            for _, curr in events.iterrows()
            if curr["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.EVAL
        ]

        # --- Structural: 3 EVAL_ROOT spans with correct names ---
        self.assertEqual(3, len(eval_roots))
        self.assertListEqual(
            ["Answer Relevance", "Context Relevance", "Groundedness"],
            sorted([
                curr["record_attributes"][SpanAttributes.EVAL_ROOT.METRIC_NAME]
                for curr in eval_roots
            ]),
        )

        # --- Selector resolution: verify correct values reached each method ---
        self.assertEqual(1, len(provider.relevance_calls))
        prompt, response = provider.relevance_calls[0]
        self.assertEqual("Who is the cutest baby in the world?", prompt)
        self.assertEqual("Kojikun!", response)

        self.assertEqual(1, len(provider.groundedness_calls))
        _, statement = provider.groundedness_calls[0]
        self.assertEqual("Kojikun!", statement)

        self.assertEqual(4, len(provider.context_relevance_calls))
        for question, context in provider.context_relevance_calls:
            self.assertEqual("Who is the cutest baby in the world?", question)
            self.assertIn(context, MockLLMProvider.CONTEXT_SCORES)

        # --- EVAL span breakdown: 4 context + 1 groundedness + 1 answer relevance ---
        self.assertEqual(6, len(evals))
        self.assertListEqual(
            [0.0, 0.25, 0.75, 0.8, 0.8, 1.0],
            sorted([
                curr["record_attributes"][SpanAttributes.EVAL.SCORE]
                for curr in evals
            ]),
        )

        # --- Aggregated EVAL_ROOT scores ---
        scores_by_name = {
            curr["record_attributes"][
                SpanAttributes.EVAL_ROOT.METRIC_NAME
            ]: curr["record_attributes"][SpanAttributes.EVAL_ROOT.SCORE]
            for curr in eval_roots
        }
        self.assertAlmostEqual(
            0.8, scores_by_name["Answer Relevance"], places=5
        )
        self.assertAlmostEqual(0.8, scores_by_name["Groundedness"], places=5)
        # mean([0.25, 1.0, 0.75, 0.0]) == 0.5
        self.assertAlmostEqual(
            0.5, scores_by_name["Context Relevance"], places=5
        )
