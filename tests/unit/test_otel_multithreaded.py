import threading
import time
from typing import Tuple

import numpy as np
from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class _TestApp:
    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    def respond_to_query(self, query: str) -> str:
        threads = []
        for _ in range(100):
            thread = threading.Thread(target=self.nested)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return get_baggage("best_baby")

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "best_baby": ret[0],
            "span_id": ret[1],
        }
    )
    def nested(self) -> Tuple[str, str]:
        time.sleep(np.random.random())
        span = trace.get_current_span()
        best_baby = get_baggage("best_baby")
        return best_baby, str(span.get_span_context().span_id)


class TestOtelMultiThreaded(OtelAppTestCase):
    def test_multithreaded(self):
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create TruApp that runs many things in parallel.
        test_app = _TestApp()
        custom_app = TruApp(test_app, main_method=test_app.respond_to_query)
        recorder = custom_app(run_name="test run", input_id="456")
        with recorder:
            recorder.attach_to_context("best_baby", "Kojikun")
            test_app.respond_to_query("test")
        # Compare results to expected.
        tru_session.force_flush()
        actual = self._get_events()
        seen_span_ids = set()
        for _, row in actual.iterrows():
            record_attributes = row["record_attributes"]
            span_type = record_attributes[f"{BASE_SCOPE}.span_type"]
            if span_type == SpanAttributes.SpanType.UNKNOWN:
                best_baby = record_attributes[f"{BASE_SCOPE}.unknown.best_baby"]
                self.assertEqual(best_baby, "Kojikun")
                span_id = record_attributes[f"{BASE_SCOPE}.unknown.span_id"]
                self.assertEqual(span_id, row["trace"]["span_id"])
                seen_span_ids.add(span_id)
            elif span_type == SpanAttributes.SpanType.MAIN:
                self.assertEqual(
                    record_attributes[f"{BASE_SCOPE}.main.main_input"],
                    "test",
                )
                self.assertEqual(
                    record_attributes[f"{BASE_SCOPE}.main.main_output"],
                    "Kojikun",
                )
        self.assertEqual(len(seen_span_ids), 100)
