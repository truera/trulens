from typing import Callable

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelRecordRoot(OtelTestCase):
    def test_overwrites_span_type(self):
        class _App:
            @instrument(span_type="obvious_question")
            def query(self, question: str) -> str:
                return "Kojikun"

        app = _App()
        tru_app = TruApp(app=app, main_method=app.query)
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("Who is the cutest baby in the world?",),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(1, len(events))
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
        )

    def test_app_specific_instrumentation(self):
        def count_wraps(func: Callable) -> int:
            if not hasattr(func, "__wrapped__"):
                return 0
            return 1 + count_wraps(func.__wrapped__)

        class _App:
            def query_a(self, question: str) -> str:
                return self.query_b(question)

            @instrument()
            def query_b(self, question: str) -> str:
                return "Kojikun"

        app1 = _App()
        TruApp(app=app1, main_method=app1.query_b)
        app2 = _App()
        app3 = _App()
        TruApp(app=app2, main_method=app2.query_a)
        TruApp(app=app3, main_method=app3.query_b)
        app4 = _App()
        TruApp(app=app4, main_method=app4.query_b)
        self.assertEqual(0, count_wraps(app1.query_a))
        self.assertEqual(1, count_wraps(app1.query_b))
        self.assertEqual(1, count_wraps(app2.query_a))
        self.assertEqual(1, count_wraps(app2.query_b))
        self.assertEqual(0, count_wraps(app3.query_a))
        self.assertEqual(1, count_wraps(app3.query_b))
        self.assertEqual(0, count_wraps(app4.query_a))
        self.assertEqual(1, count_wraps(app4.query_b))

    def test_pupr_record_root(self):
        class _App:
            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def query(self, question: str) -> str:
                return "Kojikun"

        app = _App()
        tru_app = TruApp(app=app)
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("Who is the cutest baby in the world?",),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(1, len(events))
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
        )
