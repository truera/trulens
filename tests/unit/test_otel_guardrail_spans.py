"""Tests for OTEL guardrail span instrumentation."""

from typing import List

from trulens.apps.app import TruApp
from trulens.core import Feedback
from trulens.core.guardrails.base import block_input
from trulens.core.guardrails.base import block_output
from trulens.core.guardrails.base import context_filter
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

# Simple deterministic feedback functions
_context_relevance = Feedback(
    lambda query, context: 0.0 if "irrelevant" in context else 1.0,
    name="context relevance",
)
_criminality = Feedback(
    lambda text: 1.0 if "harmful" in text else 0.0,
    name="criminality",
    higher_is_better=False,
)


def _collect_guardrail_spans(events):
    """Return all rows whose span_type is GUARDRAIL."""
    spans = []
    for _, row in events.iterrows():
        attrs = row["record_attributes"]
        if (
            attrs.get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.GUARDRAIL
        ):
            spans.append(attrs)
    return spans


class _ContextFilterApp:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            f"{SpanAttributes.RETRIEVAL.base}.return": ret
        },
    )
    @context_filter(
        feedback=_context_relevance,
        threshold=0.75,
        keyword_for_prompt="query",
    )
    def retrieve(self, query: str) -> List[str]:
        return [
            "1. This is an irrelevant comment!",
            "2. This is a relevant comment.",
            "3. This is an irrelevant comment!",
            "4. This is a relevant comment.",
        ]


class _BlockInputApp:
    @instrument()
    @block_input(
        feedback=_criminality,
        threshold=0.5,
        keyword_for_prompt="question",
        return_value="blocked",
    )
    def chat(self, question: str) -> str:
        return f"response to: {question}"


class _BlockOutputApp:
    @instrument()
    @block_output(
        feedback=_criminality,
        threshold=0.5,
        return_value="blocked",
    )
    def chat(self, question: str) -> str:
        return "harmful content" if "bad" in question else "safe response"


class TestGuardrailSpans(OtelTestCase):
    """Verify that guardrail decorators emit correctly-attributed OTEL spans."""

    def _run_app(self, app, method_name, *args):
        tru = TruApp(
            app,
            app_name="Test App",
            app_version="v1",
            main_method=getattr(app, method_name),
        )
        result = tru.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=args,
        )
        TruSession().force_flush()
        return result

    def test_context_filter_guardrail_spans(self):
        """context_filter emits one GUARDRAIL span per context with correct attrs."""
        app = _ContextFilterApp()
        result = self._run_app(app, "retrieve", "test query")
        # Only relevant contexts pass
        self.assertEqual(
            sorted(result),
            [
                "2. This is a relevant comment.",
                "4. This is a relevant comment.",
            ],
        )

        spans = _collect_guardrail_spans(self._get_events())
        self.assertEqual(
            len(spans), 4, "Expected one guardrail span per context"
        )

        for span in spans:
            self.assertEqual(
                span[SpanAttributes.GUARDRAIL.NAME], "context relevance"
            )
            self.assertIn(SpanAttributes.GUARDRAIL.SCORE, span)
            self.assertIn(SpanAttributes.GUARDRAIL.PASSED, span)
            self.assertAlmostEqual(
                span[SpanAttributes.GUARDRAIL.THRESHOLD], 0.75
            )

        passed_spans = [s for s in spans if s[SpanAttributes.GUARDRAIL.PASSED]]
        failed_spans = [
            s for s in spans if not s[SpanAttributes.GUARDRAIL.PASSED]
        ]
        self.assertEqual(len(passed_spans), 2)
        self.assertEqual(len(failed_spans), 2)

    def test_block_input_pass_emits_span(self):
        """block_input emits a GUARDRAIL span and passes safe input through."""
        app = _BlockInputApp()
        result = self._run_app(app, "chat", "safe question")
        self.assertEqual(result, "response to: safe question")

        spans = _collect_guardrail_spans(self._get_events())
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertEqual(s[SpanAttributes.GUARDRAIL.NAME], "criminality")
        self.assertAlmostEqual(s[SpanAttributes.GUARDRAIL.SCORE], 0.0)
        self.assertTrue(s[SpanAttributes.GUARDRAIL.PASSED])
        self.assertAlmostEqual(s[SpanAttributes.GUARDRAIL.THRESHOLD], 0.5)

    def test_block_input_blocked_emits_span(self):
        """block_input emits a GUARDRAIL span and returns return_value for harmful input."""
        app = _BlockInputApp()
        result = self._run_app(app, "chat", "harmful question")
        self.assertEqual(result, "blocked")

        spans = _collect_guardrail_spans(self._get_events())
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertAlmostEqual(s[SpanAttributes.GUARDRAIL.SCORE], 1.0)
        self.assertFalse(s[SpanAttributes.GUARDRAIL.PASSED])

    def test_block_output_pass_emits_span(self):
        """block_output emits a GUARDRAIL span and passes safe output through."""
        app = _BlockOutputApp()
        result = self._run_app(app, "chat", "good question")
        self.assertEqual(result, "safe response")

        spans = _collect_guardrail_spans(self._get_events())
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertEqual(s[SpanAttributes.GUARDRAIL.NAME], "criminality")
        self.assertAlmostEqual(s[SpanAttributes.GUARDRAIL.SCORE], 0.0)
        self.assertTrue(s[SpanAttributes.GUARDRAIL.PASSED])

    def test_block_output_blocked_emits_span(self):
        """block_output emits a GUARDRAIL span and returns return_value for harmful output."""
        app = _BlockOutputApp()
        result = self._run_app(app, "chat", "bad question")
        self.assertEqual(result, "blocked")

        spans = _collect_guardrail_spans(self._get_events())
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertAlmostEqual(s[SpanAttributes.GUARDRAIL.SCORE], 1.0)
        self.assertFalse(s[SpanAttributes.GUARDRAIL.PASSED])
