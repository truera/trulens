from typing import List

from trulens.apps.app import TruApp
from trulens.core import Feedback
from trulens.core.guardrails.base import context_filter
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

_context_relevance = Feedback(
    lambda query, context: 0.0 if "irrelevant" in context else 1.0,
    name="context relevance",
)


class _TestApp:
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
            "5. This is an irrelevant comment!",
        ]


class TestOtelGuardrail(OtelTestCase):
    def test_context_relevance(self) -> None:
        app = _TestApp()
        tru_recorder = TruApp(
            app, app_name="Test App", app_version="v1", main_method=app.retrieve
        )
        result = tru_recorder.instrumented_invoke_main_method(
            run_name="test run", input_id="42", main_method_args=("test",)
        )
        # Check that only relevant comments are returned.
        expected_result = [
            "2. This is a relevant comment.",
            "4. This is a relevant comment.",
        ]
        self.assertListEqual(sorted(result), expected_result)
        TruSession().force_flush()
        # Check that the span only contains the relevant comments.
        seen = False
        for _, curr in self._get_events().iterrows():
            record_attributes = curr["record_attributes"]
            return_key = f"{SpanAttributes.RETRIEVAL.base}.return"
            if return_key in record_attributes:
                self.assertFalse(seen)
                self.assertListEqual(
                    sorted(record_attributes[return_key]),
                    expected_result,
                )
                seen = True
        self.assertTrue(seen)
