from typing import Optional

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import ResourceAttributes
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
            (SpanAttributes.RECORD_ID, record_id, False),
            (ResourceAttributes.APP_NAME, "Greeter", True),
            (ResourceAttributes.APP_VERSION, "v1", True),
        ]
        if run_name:
            attribute_values.append((SpanAttributes.RUN_NAME, run_name, False))
        if input_id:
            attribute_values.append((SpanAttributes.INPUT_ID, input_id, False))
        for _, event in events.iterrows():
            for attribute, value, is_resource_attribute in attribute_values:
                if is_resource_attribute:
                    self.assertEqual(
                        value, event["resource_attributes"][attribute]
                    )
                else:
                    self.assertEqual(
                        value, event["record_attributes"][attribute]
                    )

    def test_legacy(self):
        with self._tru_recorder as recording:
            self._app.greet(name="Kojikun")
        self._validate()
        events = self._get_events()
        self.assertEqual(len(recording), 1)
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.RECORD_ID],
            recording.get().record_id,
        )
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.RECORD_ID],
            recording[0].record_id,
        )

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

    # ------------------------------------------------------------------
    # conversation_id tests
    # ------------------------------------------------------------------

    def test_conversation_id_is_set_on_spans(self):
        """conversation_id propagates to all spans when provided."""
        with self._tru_recorder(conversation_id="conv-123") as recording:
            self._app.greet(name="Kojikun")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        for _, event in events.iterrows():
            self.assertEqual(
                event["record_attributes"][SpanAttributes.CONVERSATION_ID],
                "conv-123",
            )

    def test_conversation_id_none_does_not_set_attribute(self):
        """Omitting conversation_id leaves the attribute absent from spans."""
        with self._tru_recorder(conversation_id=None) as recording:
            self._app.greet(name="Kojikun")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        for _, event in events.iterrows():
            self.assertNotIn(
                SpanAttributes.CONVERSATION_ID, event["record_attributes"]
            )

    def test_legacy_context_manager_no_conversation_id(self):
        """Plain `with tru_app as recording:` works and has no conversation_id."""
        with self._tru_recorder as recording:
            self._app.greet(name="Kojikun")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        for _, event in events.iterrows():
            self.assertNotIn(
                SpanAttributes.CONVERSATION_ID, event["record_attributes"]
            )

    def test_multiple_invocations_same_conversation_id(self):
        """Two separate context managers with the same conversation_id both carry it."""
        conv_id = "multi-turn-conv"

        with self._tru_recorder(conversation_id=conv_id):
            self._app.greet(name="Turn1")

        TruSession().force_flush()
        first_events = self._get_events()

        with self._tru_recorder(conversation_id=conv_id):
            self._app.greet(name="Turn2")

        TruSession().force_flush()
        all_events = self._get_events()

        # First batch carries the conversation_id.
        for _, event in first_events.iterrows():
            self.assertEqual(
                event["record_attributes"][SpanAttributes.CONVERSATION_ID],
                conv_id,
            )

        # Second batch (new rows) also carries it.
        second_events = all_events.iloc[len(first_events) :]
        for _, event in second_events.iterrows():
            self.assertEqual(
                event["record_attributes"][SpanAttributes.CONVERSATION_ID],
                conv_id,
            )
