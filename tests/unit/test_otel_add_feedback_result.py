from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelAddFeedbackResult(OtelTestCase):
    def test_add_single_feedback_result(self):
        tru_session = TruSession()

        # Create a simple app.
        class SimpleApp:
            @instrument()
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = SimpleApp()
        tru_app = TruApp(app, app_name="TestApp", app_version="v1")

        # Invoke and record.
        with tru_app as recording:
            app.greet("Nolan")
            app.greet("Kojikun")
            app.greet("Sachiboy")

        # Add the feedback result.
        self.assertEqual(len(recording), 3)
        record = recording[1]
        tru_session.add_feedback_result(
            record=record,
            feedback_name="human_thumbs_up",
            feedback_result=1,
            higher_is_better=True,
        )
        tru_session.force_flush()

        # Check events.
        events = self._get_events()
        self.assertListEqual(
            [
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.EVAL_ROOT,
                SpanAttributes.SpanType.EVAL,
            ],
            events["record_attributes"]
            .apply(lambda curr: curr[SpanAttributes.SPAN_TYPE])
            .tolist(),
        )

        # Check the record with the feedback result.
        events = events.iloc[[1, 3, 4]]
        record_attributes = events["record_attributes"]
        self.assertListEqual(
            [
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.EVAL_ROOT,
                SpanAttributes.SpanType.EVAL,
            ],
            record_attributes.apply(
                lambda curr: curr[SpanAttributes.SPAN_TYPE]
            ).tolist(),
        )
        for attribute in [
            SpanAttributes.RECORD_ID,
            SpanAttributes.RUN_NAME,
            SpanAttributes.INPUT_ID,
        ]:
            self.assertListEqual(
                3
                * [record_attributes.apply(lambda x: x.get(attribute)).iloc[0]],
                record_attributes.apply(lambda x: x.get(attribute)).tolist(),
            )
        self.assertEqual(
            record_attributes.iloc[1][SpanAttributes.EVAL_ROOT.METRIC_NAME],
            "human_thumbs_up",
        )
        self.assertEqual(
            record_attributes.iloc[1][SpanAttributes.EVAL_ROOT.SCORE], 1
        )
