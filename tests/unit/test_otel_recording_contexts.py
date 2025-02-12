from trulens.apps.app import TruApp
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class _TestApp:
    def greet(self, *, name: str) -> str:
        return f"Hello, {name}!"


class TestOtelRecordingContexts(OtelAppTestCase):
    def setUp(self):
        super().setUp()
        self._app = _TestApp()
        self._tru_recorder = TruApp(
            self._app,
            app_name="Greeter",
            app_version="v1",
            main_method=self._app.greet,
        )

    def test_legacy(self):
        with self._tru_recorder:
            self._app.greet(name="Kojikun")
        # TODO(this_pr): everything below should be put in a helper function.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        record_id = events.iloc[0]["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        for _, event in events.iterrows():
            for attribute, value in [
                (SpanAttributes.RECORD_ID, record_id),
                (SpanAttributes.APP_NAME, "Greeter"),
                (SpanAttributes.APP_VERSION, "v1"),
                # (SpanAttributes.RUN_NAME, ""), # TODO(this_pr): confirm it's okay they're not there!
                # (SpanAttributes.INPUT_ID, ""), # TODO(this_pr): confirm it's okay they're not there!
            ]:
                self.assertEqual(value, event["record_attributes"][attribute])

    def test_new(self):
        with self._tru_recorder.run("test_run"):
            with self._tru_recorder.input("42"):
                self._app.greet(name="Kojikun")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        record_id = events.iloc[0]["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        for _, event in events.iterrows():
            for attribute, value in [
                (SpanAttributes.RECORD_ID, record_id),
                (SpanAttributes.APP_NAME, "Greeter"),
                (SpanAttributes.APP_VERSION, "v1"),
                (SpanAttributes.RUN_NAME, "test_run"),
                (SpanAttributes.INPUT_ID, "42"),
            ]:
                self.assertEqual(value, event["record_attributes"][attribute])

    def test_instrumented_invoke_main_method(self):
        self._tru_recorder.instrumented_invoke_main_method(
            "test_run", "42", main_method_kwargs={"name": "Kojikun"}
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        record_id = events.iloc[0]["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        for _, event in events.iterrows():
            for attribute, value in [
                (SpanAttributes.RECORD_ID, record_id),
                (SpanAttributes.APP_NAME, "Greeter"),
                (SpanAttributes.APP_VERSION, "v1"),
                (SpanAttributes.RUN_NAME, "test_run"),
                (SpanAttributes.INPUT_ID, "42"),
            ]:
                self.assertEqual(value, event["record_attributes"][attribute])
