import pandas as pd
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class Greeter:
    def __init__(self):
        self.best_baby_finder = BestBabyFinder()

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def greet(self):
        return f"Hello, {self.best_baby_finder.find_best_baby()}!"


class BestBabyFinder:
    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def find_best_baby(self) -> str:
        return "Kojikun"


class TestOtelNestedApps(OtelAppTestCase):
    def test_outer_invoked_and_inner_invoked(self):
        # Create and run app.
        app = Greeter()
        outer_tru_app = TruApp(app, app_name="greeter", app_version="v1")
        inner_tru_app = TruApp(
            app.best_baby_finder, app_name="Kojikun", app_version="v2"
        )
        with outer_tru_app:
            with inner_tru_app:
                app.greet()
        # Get events in timestamp order breaking ties by app name.
        TruSession().force_flush()
        events = self._get_events()
        record_attribute_to_app_name = lambda curr: curr[
            SpanAttributes.APP_NAME
        ]
        events.sort_values(
            by=["start_timestamp", "record_attributes"],
            key=lambda ser: ser
            if isinstance(ser.iloc[0], pd.Timestamp)
            else ser.apply(record_attribute_to_app_name),
        )
        self.assertEqual(len(events), 3)
        # Verify span types.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        self.assertEqual(
            events.iloc[1]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.NESTED_RECORD_ROOT,
        )
        self.assertEqual(
            events.iloc[2]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        # Verify record ids.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.RECORD_ID],
            events.iloc[1]["record_attributes"][SpanAttributes.RECORD_ID],
        )
        self.assertNotEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.RECORD_ID],
            events.iloc[2]["record_attributes"][SpanAttributes.RECORD_ID],
        )
        # Verify app names.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.APP_NAME],
            "greeter",
        )
        self.assertEqual(
            events.iloc[1]["record_attributes"][SpanAttributes.APP_NAME],
            "greeter",
        )
        self.assertEqual(
            events.iloc[2]["record_attributes"][SpanAttributes.APP_NAME],
            "Kojikun",
        )
        # Verify trace ids.
        self.assertEqual(
            events.iloc[0]["trace"]["trace_id"],
            events.iloc[1]["trace"]["trace_id"],
        )
        self.assertEqual(
            events.iloc[0]["trace"]["trace_id"],
            events.iloc[2]["trace"]["trace_id"],
        )

    def test_outer_invoked_inner_not_invoked(self):
        # Create and run app.
        app = Greeter()
        outer_tru_app = TruApp(app, app_name="greeter", app_version="v1")
        TruApp(app.best_baby_finder, app_name="Kojikun", app_version="v2")
        with outer_tru_app:
            app.greet()
        # Get events in timestamp order breaking ties by app name.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2)
        # Verify span types.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        self.assertEqual(
            events.iloc[1]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.NESTED_RECORD_ROOT,
        )
        # Verify record ids.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.RECORD_ID],
            events.iloc[1]["record_attributes"][SpanAttributes.RECORD_ID],
        )
        # Verify app names.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.APP_NAME],
            "greeter",
        )
        self.assertEqual(
            events.iloc[1]["record_attributes"][SpanAttributes.APP_NAME],
            "greeter",
        )
        # Verify trace ids.
        self.assertEqual(
            events.iloc[0]["trace"]["trace_id"],
            events.iloc[1]["trace"]["trace_id"],
        )

    def test_outer_not_invoked_inner_invoked(self):
        # Create and run app.
        app = Greeter()
        TruApp(app, app_name="greeter", app_version="v1")
        inner_tru_app = TruApp(
            app.best_baby_finder, app_name="Kojikun", app_version="v2"
        )
        with inner_tru_app:
            app.greet()
        # Get events in timestamp order breaking ties by app name.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 1)
        # Verify span types.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        # Verify app names.
        self.assertEqual(
            events.iloc[0]["record_attributes"][SpanAttributes.APP_NAME],
            "Kojikun",
        )
