from typing import Optional

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _TestApp:
    def greet(self, *, name: str) -> str:
        return self.capitalize(f"Hello, {name}!")

    @instrument()
    def capitalize(self, s: str) -> str:
        return s.upper()


class TestOtelRecordingContexts(OtelTestCase):
    def setUp(self):
        super().setUp()
        self._app = _TestApp()
        self._tru_recorder = TruApp(
            self._app,
            app_name="Greeter",
            app_version="v1",
            main_method=self._app.greet,
        )

    def _validate(
        self, run_name: Optional[str] = None, input_id: Optional[str] = None
    ):
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        record_id = events.iloc[0]["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        attribute_values = [
            (SpanAttributes.RECORD_ID, record_id),
            (SpanAttributes.APP_NAME, "Greeter"),
            (SpanAttributes.APP_VERSION, "v1"),
        ]
        if run_name:
            attribute_values.append((SpanAttributes.RUN_NAME, run_name))
        if input_id:
            attribute_values.append((SpanAttributes.INPUT_ID, input_id))
        for _, event in events.iterrows():
            for attribute, value in attribute_values:
                self.assertEqual(value, event["record_attributes"][attribute])

    def test_legacy(self):
        with self._tru_recorder:
            self._app.greet(name="Kojikun")
        self._validate()

    def test_new(self):
        with self._tru_recorder.run("test_run"):
            with self._tru_recorder.input("42"):
                self._app.greet(name="Kojikun")
        self._validate("test_run", "42")

    def test_instrumented_invoke_main_method(self):
        self._tru_recorder.instrumented_invoke_main_method(
            "test_run", "42", main_method_kwargs={"name": "Kojikun"}
        )
        self._validate("test_run", "42")
