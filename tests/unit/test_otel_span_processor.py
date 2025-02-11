from opentelemetry import trace
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class _TestApp:
    @instrument()
    def greet(self, name: str) -> str:
        ret = f"Hello, {name}!"
        ret = self.capitalize(ret)
        return ret

    def capitalize(self, txt: str) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("capitalize") as span:
            span.set_attribute("best_baby", "Kojikun")
        return txt.upper()


class TestOtelSpanProcessor(OtelAppTestCase):
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
        with tru_recorder(run_name="test run", input_id="42"):
            app.greet("Kojikun")
        # Verify.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)
        self.assertEqual(
            "Kojikun", events.iloc[-1]["record_attributes"]["best_baby"]
        )
        for curr in [
            SpanAttributes.RECORD_ID,
            SpanAttributes.APP_NAME,
            SpanAttributes.APP_VERSION,
            SpanAttributes.RUN_NAME,
            SpanAttributes.INPUT_ID,
        ]:
            self.assertEqual(
                events.iloc[1]["record_attributes"][curr],
                events.iloc[-1]["record_attributes"][curr],
            )
