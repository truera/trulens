from typing import List
import unittest

from trulens.apps.custom import TruCustomApp
from trulens.core import Feedback
from trulens.core.guardrails.base import context_filter
from trulens.experimental.otel_tracing.core.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase

_context_relevance = Feedback(
    lambda query, context: 0.0 if "irrelevant" in context else 1.0,
    name="context relevance",
)


class _TestApp:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {"return": ret},
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


class TestOtelGuardrail(OtelAppTestCase):
    def test_context_relevance(self) -> None:
        app = _TestApp()
        tru_recorder = TruCustomApp(app, app_name="Test App", app_version="v1")
        with tru_recorder(run_name="test run", input_id="42"):
            result = app.retrieve("test")
        # Check that only relevant comments are returned.
        expected_result = [
            "2. This is a relevant comment.",
            "4. This is a relevant comment.",
        ]
        self.assertListEqual(sorted(result), expected_result)
        # Check that the span only contains the relevant comments.
        seen = False
        for _, curr in self._get_events().iterrows():
            record_attributes = curr["RecordAttributes"]
            return_key = f"{SpanAttributes.RETRIEVAL.base}.return"
            if return_key in record_attributes:
                self.assertFalse(seen)
                self.assertListEqual(
                    sorted(curr["RecordAttributes"][return_key]),
                    expected_result,
                )
                seen = True
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
