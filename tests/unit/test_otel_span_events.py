"""
Tests for persisting and selecting OTEL span events.

GenAI message content (`gen_ai.input.messages`, `gen_ai.output.messages`) is
emitted as a span *event* rather than a span attribute, and only when content
capture is opted in. These tests cover the round trip: emission, persistence,
and selection by a metric.
"""

import pytest
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.span import set_content_capture
from trulens.otel.semconv.trace import GenAIEvents
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

_QUESTION = "Who is the best baby?"


class _GenerationApp:
    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes=lambda ret, exception, *args, **kwargs: {
            "prompt": kwargs.get("question"),
            "completion": ret,
        },
    )
    def generate(self, question: str) -> str:
        return "Kojikun"

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "question",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def answer(self, question: str) -> str:
        return self.generate(question=question)


def _echo_input_messages(messages: str) -> float:
    return 1.0 if messages == _QUESTION else 0.0


@pytest.mark.optional
class TestOtelSpanEvents(OtelTestCase):
    def _record(self, app_name: str, feedbacks=None) -> TruApp:
        app = _GenerationApp()
        tru_app = TruApp(
            app,
            app_name=app_name,
            app_version="v1",
            main_method=app.answer,
            feedbacks=feedbacks or [],
        )
        with tru_app:
            app.answer(_QUESTION)
        TruSession().force_flush()
        return tru_app

    def test_span_events_are_persisted_when_content_capture_is_on(self) -> None:
        set_content_capture(True)
        self.addCleanup(set_content_capture, None)

        self._record("Span Event App")

        events = self._get_events()
        with_events = [
            curr["record"]
            for _, curr in events.iterrows()
            if "events" in curr["record"]
        ]
        # Only the generation span emits an inference event.
        self.assertEqual(1, len(with_events))
        self.assertEqual(
            GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
            with_events[0]["events"][0]["name"],
        )
        attributes = with_events[0]["events"][0]["attributes"]
        self.assertEqual(
            _QUESTION, attributes[GenAIEvents.EventAttributes.INPUT_MESSAGES]
        )
        self.assertEqual(
            "Kojikun",
            attributes[GenAIEvents.EventAttributes.OUTPUT_MESSAGES],
        )

    def test_no_events_recorded_when_content_capture_is_off(self) -> None:
        # Content capture is opt-in, so the default must leave persisted rows
        # untouched: no `events` key at all.
        set_content_capture(False)
        self.addCleanup(set_content_capture, None)

        self._record("No Span Event App")

        events = self._get_events()
        self.assertGreater(len(events), 0)
        for _, curr in events.iterrows():
            self.assertNotIn("events", curr["record"])

    def test_metric_can_select_a_span_event_attribute(self) -> None:
        set_content_capture(True)
        self.addCleanup(set_content_capture, None)

        metric = Metric(
            implementation=_echo_input_messages,
            name="input_messages",
            selectors={
                "messages": Selector(
                    span_type=SpanAttributes.SpanType.GENERATION,
                    span_event_attribute=GenAIEvents.EventAttributes.INPUT_MESSAGES,
                    span_event_name=GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS,
                ),
            },
        )

        tru_app = self._record("Span Event Metric App", feedbacks=[metric])
        tru_app.compute_feedbacks()
        TruSession().force_flush()

        eval_roots = [
            curr["record_attributes"]
            for _, curr in self._get_events().iterrows()
            if curr["record_attributes"][SpanAttributes.SPAN_TYPE]
            == SpanAttributes.SpanType.EVAL_ROOT
        ]
        self.assertEqual(1, len(eval_roots))
        self.assertEqual(1.0, eval_roots[0][SpanAttributes.EVAL_ROOT.SCORE])
        # Provenance must show the value came from an event, so it is
        # distinguishable from a span attribute of the same name.
        self.assertEqual(
            f"{GenAIEvents.CLIENT_INFERENCE_OPERATION_DETAILS}/"
            f"{GenAIEvents.EventAttributes.INPUT_MESSAGES}",
            eval_roots[0][
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE + ".messages"
            ],
        )
