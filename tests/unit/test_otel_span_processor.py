from opentelemetry import trace
from trulens.apps.app import TruApp
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _TestApp:
    def greet(self, name: str) -> str:
        ret = f"Hello, {name}!"
        ret = self.capitalize(ret)
        return ret

    def capitalize(self, txt: str) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("capitalize") as span:
            span.set_attribute("best_baby", "Kojikun")
        return txt.upper()


class TestOtelSpanProcessor(OtelTestCase):
    def test_span_processor(self) -> None:
        # Create app.
        app = _TestApp()
        tru_recorder = TruApp(
            app,
            app_name="Simple Greeter",
            app_version="v1",
            main_method=app.greet,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run", input_id="42", main_method_args=("Kojikun",)
        )
        # Invoke without recording.
        app.greet("Nolan")
        # Verify.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(
            "Kojikun", events.iloc[1]["record_attributes"]["best_baby"]
        )
        for curr in [
            SpanAttributes.RECORD_ID,
            SpanAttributes.APP_NAME,
            SpanAttributes.APP_VERSION,
            SpanAttributes.RUN_NAME,
            SpanAttributes.INPUT_ID,
        ]:
            self.assertTrue(bool(events.iloc[1]["record_attributes"][curr]))
            self.assertEqual(
                events.iloc[0]["record_attributes"][curr],
                events.iloc[1]["record_attributes"][curr],
            )
