from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelRecordRoot(OtelTestCase):
    def test_no_main_method(self):
        class App:
            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def main(self):
                pass

            def main2(self):
                pass

        app = App()
        tru_app = TruApp(app, app_name="test", app_version="v1")
        self.assertEqual(tru_app.main_method_name, "main")
        # Record and invoke.
        tru_app.instrumented_invoke_main_method(run_name="1", input_id="42")
        TruSession().force_flush()
        self.assertEqual(len(self._get_events()), 1)

    def test_main_method_with_same_record_root_span(self):
        class App:
            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def main(self):
                pass

            def main2(self):
                pass

        app = App()
        tru_app = TruApp(
            app, main_method=app.main, app_name="test", app_version="v1"
        )
        self.assertEqual(tru_app.main_method_name, "main")
        # Record and invoke.
        tru_app.instrumented_invoke_main_method(run_name="1", input_id="42")
        TruSession().force_flush()
        self.assertEqual(len(self._get_events()), 1)

    def test_main_method_with_different_record_root_span(self):
        class App:
            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def main(self):
                pass

            def main2(self):
                pass

        app = App()
        with self.assertRaisesRegex(
            ValueError,
            "Must have exactly one main method or method decorated with span type 'record_root'! Found: ",
        ):
            TruApp(
                app, main_method=app.main2, app_name="test", app_version="v1"
            )

    def test_multiple_record_root_spans(self):
        class App:
            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def main(self):
                pass

            @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
            def main2(self):
                pass

        app = App()
        with self.assertRaisesRegex(
            ValueError,
            "Must have exactly one main method or method decorated with span type 'record_root'! Found: ",
        ):
            TruApp(app, app_name="test", app_version="v1")
