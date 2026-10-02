from typing import List

import numpy as np
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.exporter import (
    utils as exporter_utils,
)
from trulens.experimental.otel_tracing.core.session import (
    _set_up_tracer_provider,
)
from trulens.feedback.computer import NON_NUMERIC_SCORE_ERROR_TYPE
from trulens.otel.semconv.trace import ErrorAttributes
from trulens.otel.semconv.trace import GenAIEvents
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
        f_groundedness = Metric(
            implementation=mock_groundedness,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        )
        f_answer_relevance = Metric(
            implementation=mock_answer_relevance,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        )
        f_context_relevance = Metric(
            implementation=mock_context_relevance,
            name="Context Relevance",
            agg=np.mean,
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
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


_EVENT_ATTRIBUTES = GenAIEvents.EventAttributes
_GROUNDEDNESS_REASON = "The answer is supported by the second context."


def _create_rag_triad_app() -> TruApp:
    """RAG triad app whose metrics return fixed scores.

    Groundedness returns a reason, so its event must carry an explanation.
    Context Relevance is aggregated over four contexts, so its event must not.
    """
    context_relevance_results = [0.25, 1, 0.75, 0]

    def mock_groundedness(source: List[str], statement: str):
        return 0.99, {"reason": _GROUNDEDNESS_REASON}

    def mock_answer_relevance(prompt: str, response: str) -> float:
        return 0.98

    def mock_context_relevance(question: str, context: str) -> float:
        return context_relevance_results.pop(0)

    metrics = [
        Metric(
            implementation=mock_context_relevance,
            name="Context Relevance",
            agg=np.mean,
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
        ),
        Metric(
            implementation=mock_groundedness,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        ),
        Metric(
            implementation=mock_answer_relevance,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        ),
    ]
    app = _RAGApp()
    tru_app = TruApp(
        app,
        app_name="RAG Triad Event App",
        app_version="v1",
        feedbacks=metrics,
    )
    tru_app.stop_evaluator()
    with tru_app:
        app.query("Who is the cutest baby in the world?")
    TruSession().force_flush()
    return tru_app


def _evaluation_result_events(events) -> List[dict]:
    return [
        event
        for event in events
        if event["name"] == GenAIEvents.EVALUATION_RESULT
    ]


def _create_single_metric_app(app_name: str, metric: Metric) -> TruApp:
    app = _RAGApp()
    tru_app = TruApp(
        app,
        app_name=app_name,
        app_version="v1",
        feedbacks=[metric],
    )
    tru_app.stop_evaluator()
    with tru_app:
        app.query("Who is the cutest baby in the world?")
    TruSession().force_flush()
    return tru_app


class _BadStr:
    """Object whose `__str__` raises, like a broken judge result."""

    def __str__(self) -> str:
        raise RuntimeError("boom")


@pytest.mark.optional
class TestOtelEvaluationResultEvent(OtelTestCase):
    exporter: InMemorySpanExporter
    span_processor: SimpleSpanProcessor

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Capture the spans as the SDK hands them to any exporter, next to the
        # TruLens database exporter that the session already uses. The SDK
        # cannot remove a span processor, so add one for the whole class.
        cls.exporter = InMemorySpanExporter()
        cls.span_processor = SimpleSpanProcessor(cls.exporter)
        _set_up_tracer_provider().add_span_processor(cls.span_processor)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.span_processor.shutdown()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.exporter.clear()

    def _finished_eval_root_spans(self) -> list:
        return [
            span
            for span in self.exporter.get_finished_spans()
            if span.attributes.get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.EVAL_ROOT
        ]

    def test_rag_triad_emits_one_event_per_metric_result(self) -> None:
        tru_app = _create_rag_triad_app()
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        eval_roots = self._finished_eval_root_spans()
        self.assertEqual(3, len(eval_roots))
        by_name = {}
        for span in eval_roots:
            events = _evaluation_result_events(
                {"name": e.name, "attributes": dict(e.attributes)}
                for e in span.events
            )
            self.assertEqual(1, len(events))
            attributes = events[0]["attributes"]
            metric_name = span.attributes[SpanAttributes.EVAL_ROOT.METRIC_NAME]
            self.assertEqual(
                metric_name, attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME]
            )
            self.assertEqual(
                span.attributes[SpanAttributes.EVAL_ROOT.SCORE],
                attributes[_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE],
            )
            self.assertNotIn(
                _EVENT_ATTRIBUTES.EVALUATION_SCORE_LABEL, attributes
            )
            self.assertNotIn(ErrorAttributes.TYPE, attributes)
            by_name[metric_name] = attributes
        self.assertEqual(
            {
                "Answer Relevance": 0.98,
                "Context Relevance": 0.5,
                "Groundedness": 0.99,
            },
            {
                name: attributes[_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE]
                for name, attributes in by_name.items()
            },
        )
        self.assertEqual(
            _GROUNDEDNESS_REASON,
            by_name["Groundedness"][_EVENT_ATTRIBUTES.EVALUATION_EXPLANATION],
        )
        # Aggregated over children, so no single explanation applies.
        self.assertNotIn(
            _EVENT_ATTRIBUTES.EVALUATION_EXPLANATION,
            by_name["Context Relevance"],
        )
        self.assertNotIn(
            _EVENT_ATTRIBUTES.EVALUATION_EXPLANATION,
            by_name["Answer Relevance"],
        )

        # The EVAL spans are unchanged: one per (sub-)evaluation.
        evals = [
            curr
            for _, curr in self._get_events().iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL
        ]
        self.assertEqual(6, len(evals))
        for _, curr in self._get_events().iterrows():
            if (
                curr["record_attributes"][SpanAttributes.SPAN_TYPE]
                != SpanAttributes.SpanType.EVAL_ROOT
            ):
                self.assertEqual(
                    [],
                    _evaluation_result_events(curr["record"].get("events", [])),
                )

    def test_event_is_stored_in_the_database(self) -> None:
        tru_app = _create_rag_triad_app()
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        eval_roots = [
            curr
            for _, curr in self._get_events().iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        self.assertEqual(3, len(eval_roots))
        for curr in eval_roots:
            events = _evaluation_result_events(curr["record"]["events"])
            self.assertEqual(1, len(events))
            attributes = events[0]["attributes"]
            self.assertEqual(
                curr["record_attributes"][SpanAttributes.EVAL_ROOT.METRIC_NAME],
                attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME],
            )
            self.assertEqual(
                curr["record_attributes"][SpanAttributes.EVAL_ROOT.SCORE],
                attributes[_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE],
            )

    def test_event_is_in_the_snowflake_span_proto(self) -> None:
        tru_app = _create_rag_triad_app()
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        eval_roots = self._finished_eval_root_spans()
        self.assertEqual(3, len(eval_roots))
        for span in eval_roots:
            proto = exporter_utils.convert_readable_span_to_proto(span)
            events = [
                e
                for e in proto.events
                if e.name == GenAIEvents.EVALUATION_RESULT
            ]
            self.assertEqual(1, len(events))
            attributes = {kv.key: kv.value for kv in events[0].attributes}
            self.assertEqual(
                span.attributes[SpanAttributes.EVAL_ROOT.METRIC_NAME],
                attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME].string_value,
            )
            self.assertEqual(
                span.attributes[SpanAttributes.EVAL_ROOT.SCORE],
                attributes[
                    _EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE
                ].double_value,
            )

    def test_event_is_in_the_otlp_export(self) -> None:
        trace_encoder = pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.common.trace_encoder"
        )
        tru_app = _create_rag_triad_app()
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        eval_roots = self._finished_eval_root_spans()
        self.assertEqual(3, len(eval_roots))
        # The OTLP span exporters serialise spans with this encoder.
        request = trace_encoder.encode_spans(eval_roots)
        spans = [
            span
            for resource_spans in request.resource_spans
            for scope_spans in resource_spans.scope_spans
            for span in scope_spans.spans
        ]
        self.assertEqual(3, len(spans))
        scores = {}
        for span in spans:
            events = [
                e
                for e in span.events
                if e.name == GenAIEvents.EVALUATION_RESULT
            ]
            self.assertEqual(1, len(events))
            attributes = {kv.key: kv.value for kv in events[0].attributes}
            scores[
                attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME].string_value
            ] = attributes[
                _EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE
            ].double_value
        self.assertEqual(
            {
                "Answer Relevance": 0.98,
                "Context Relevance": 0.5,
                "Groundedness": 0.99,
            },
            scores,
        )

    def test_failed_metric_sets_error_type_and_no_score(self) -> None:
        def failing_metric(prompt: str, response: str) -> float:
            raise ValueError("judge unavailable")

        tru_app = _create_single_metric_app(
            "RAG Failing Metric App",
            Metric(
                implementation=failing_metric,
                name="Answer Relevance",
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            ),
        )
        tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
        TruSession().force_flush()

        eval_roots = self._finished_eval_root_spans()
        self.assertEqual(1, len(eval_roots))
        events = _evaluation_result_events(
            {"name": e.name, "attributes": dict(e.attributes)}
            for e in eval_roots[0].events
        )
        self.assertEqual(1, len(events))
        attributes = events[0]["attributes"]
        self.assertEqual(
            "Answer Relevance", attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME]
        )
        self.assertEqual("ValueError", attributes[ErrorAttributes.TYPE])
        self.assertNotIn(_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE, attributes)
        self.assertNotIn(_EVENT_ATTRIBUTES.EVALUATION_EXPLANATION, attributes)

    def _single_event_attributes(self) -> dict:
        eval_roots = self._finished_eval_root_spans()
        self.assertEqual(1, len(eval_roots))
        self.assertNotIn(
            SpanAttributes.EVAL_ROOT.ERROR, eval_roots[0].attributes
        )
        events = _evaluation_result_events(
            {"name": e.name, "attributes": dict(e.attributes)}
            for e in eval_roots[0].events
        )
        self.assertEqual(1, len(events))
        return events[0]["attributes"]

    def test_explanation_with_broken_str_does_not_fail_metric(self) -> None:
        def metric_with_bad_reason(prompt: str, response: str):
            return 0.7, {"reason": _BadStr()}

        tru_app = _create_single_metric_app(
            "RAG Bad Reason App",
            Metric(
                implementation=metric_with_bad_reason,
                name="Answer Relevance",
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            ),
        )
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        for span in self.exporter.get_finished_spans():
            self.assertNotIn(SpanAttributes.EVAL.ERROR, span.attributes)
        attributes = self._single_event_attributes()
        self.assertEqual(
            0.7, attributes[_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE]
        )
        self.assertEqual(
            "<_BadStr>", attributes[_EVENT_ATTRIBUTES.EVALUATION_EXPLANATION]
        )
        self.assertNotIn(ErrorAttributes.TYPE, attributes)

    def test_non_numeric_aggregate_sets_error_type(self) -> None:
        context_relevance_results = [0.25, 1, 0.75, 0]

        def mock_context_relevance(question: str, context: str) -> float:
            return context_relevance_results.pop(0)

        tru_app = _create_single_metric_app(
            "RAG Non Numeric Aggregate App",
            Metric(
                implementation=mock_context_relevance,
                name="Context Relevance",
                agg=lambda scores: "not a number",
                selectors={
                    "question": Selector.select_record_input(),
                    "context": Selector.select_context(collect_list=False),
                },
            ),
        )
        tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
        TruSession().force_flush()

        attributes = self._single_event_attributes()
        self.assertEqual(
            "Context Relevance", attributes[_EVENT_ATTRIBUTES.EVALUATION_NAME]
        )
        self.assertEqual(
            NON_NUMERIC_SCORE_ERROR_TYPE, attributes[ErrorAttributes.TYPE]
        )
        self.assertNotIn(_EVENT_ATTRIBUTES.EVALUATION_SCORE_VALUE, attributes)
