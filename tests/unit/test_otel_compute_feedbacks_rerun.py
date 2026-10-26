"""Rerunning `compute_feedbacks` on events whose feedbacks are computed."""

from typing import Callable, List, Tuple

from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestComputeFeedbacksRerun(OtelTestCase):
    @staticmethod
    def _create_greeter(feedbacks: List[Metric]) -> Tuple[Callable, TruApp]:
        class _Greeter:
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = _Greeter()
        tru_app = TruApp(
            app,
            app_name="Greeter",
            app_version="v1",
            main_method=app.greet,
            feedbacks=feedbacks,
            start_evaluator=False,
        )
        return app.greet, tru_app

    @staticmethod
    def _record_input_metric(name: str, calls: List[str]) -> Metric:
        def implementation(input: str) -> float:
            calls.append(input)
            return 1.0

        return Metric(
            implementation=implementation,
            name=name,
            selectors={
                "input": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            },
        )

    def _get_eval_root_record_ids(self) -> List[str]:
        return [
            curr["record_attributes"][SpanAttributes.RECORD_ID]
            for _, curr in self._get_events().iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]

    def test_compute_feedbacks_rerun_on_computed_events_does_not_raise(
        self,
    ) -> None:
        calls = []
        greet, tru_app = self._create_greeter([
            self._record_input_metric("input_metric", calls)
        ])
        with tru_app:
            greet("Kojikun")
        with tru_app:
            greet("Nolan")
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        self.assertEqual(2, len(self._get_eval_root_record_ids()))
        num_events = len(self._get_events())
        # Every feedback is already computed, so this has nothing left to do
        # and must not raise even though `raise_error_on_no_feedbacks_computed`
        # defaults to True.
        with self.assertLogs("trulens.feedback.computer", level="INFO") as logs:
            tru_app.compute_feedbacks()
        TruSession().force_flush()
        self.assertEqual(num_events, len(self._get_events()))
        self.assertEqual(["Kojikun", "Nolan"], sorted(calls))
        self.assertTrue(
            any("nothing left to compute" in curr for curr in logs.output)
        )

    def test_compute_feedbacks_rerun_computes_only_missing_feedbacks(
        self,
    ) -> None:
        calls = []
        greet, tru_app = self._create_greeter([
            self._record_input_metric("input_metric", calls)
        ])
        with tru_app as recording:
            greet("Kojikun")
        record_id_1 = recording.get().record_id
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        with tru_app as recording:
            greet("Nolan")
        record_id_2 = recording.get().record_id
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        self.assertEqual(["Kojikun", "Nolan"], calls)
        self.assertEqual(
            sorted([record_id_1, record_id_2]),
            sorted(self._get_eval_root_record_ids()),
        )

    def test_compute_feedbacks_rerun_still_raises_when_no_inputs_match(
        self,
    ) -> None:
        calls = []
        unmatched_calls = []
        unmatched_metric = Metric(
            implementation=lambda input: unmatched_calls.append(input) or 1.0,
            name="unmatched_metric",
            selectors={
                "input": Selector(
                    span_name="no.such.span",
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            },
        )
        greet, tru_app = self._create_greeter([
            self._record_input_metric("input_metric", calls),
            unmatched_metric,
        ])
        with tru_app:
            greet("Kojikun")
        TruSession().force_flush()
        # A selector that matches no spans is a misconfiguration, and it
        # must keep raising both on the first run and on a rerun where the
        # other metric has nothing left to compute.
        for _ in range(2):
            with self.assertRaisesRegex(
                ValueError, "^No feedbacks were computed!$"
            ):
                tru_app.compute_feedbacks()
            TruSession().force_flush()
        self.assertEqual(["Kojikun"], calls)
        self.assertEqual([], unmatched_calls)
        self.assertEqual(1, len(self._get_eval_root_record_ids()))

    def test_compute_feedbacks_rerun_on_computed_conversations_does_not_raise(
        self,
    ) -> None:
        calls = []

        def conversation_score(records: list) -> float:
            calls.append(records)
            return 1.0

        greet, tru_app = self._create_greeter([
            Metric(
                implementation=conversation_score,
                name="conversation_score",
            ).on_conversation()
        ])
        with tru_app(conversation_id="conversation-1"):
            greet("Kojikun")
        with tru_app(conversation_id="conversation-1"):
            greet("Nolan")
        TruSession().force_flush()
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        self.assertEqual(1, len(self._get_eval_root_record_ids()))
        num_events = len(self._get_events())
        tru_app.compute_feedbacks()
        TruSession().force_flush()
        self.assertEqual(num_events, len(self._get_events()))
        self.assertEqual(1, len(calls))
