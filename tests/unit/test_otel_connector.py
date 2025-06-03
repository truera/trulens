from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelConnector(OtelTestCase):
    def test_get_events(self) -> None:
        class SimpleApp:
            @instrument()
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = SimpleApp()
        tru_app = TruApp(app, app_name="SimpleApp", app_version="v1")
        with tru_app as recording:
            app.greet("Kojikun")
            app.greet("Nolan")
            app.greet("Sachiboy")
        kojikun_record_id = recording[0].record_id
        nolan_record_id = recording[1].record_id
        sachiboy_record_id = recording[2].record_id
        connector = TruSession().connector
        # Test no record id.
        res = connector.get_events(app_id=tru_app.app_id)
        self.assertEqual(len(res), 3)
        self.assertEqual(
            {kojikun_record_id, nolan_record_id, sachiboy_record_id},
            set(
                res["record_attributes"].apply(
                    lambda curr: curr.get(SpanAttributes.RECORD_ID)
                )
            ),
        )
        # Test one record id.
        res = connector.get_events(record_ids=[nolan_record_id])
        self.assertEqual(len(res), 1)
        self.assertEqual(
            nolan_record_id,
            res.iloc[0]["record_attributes"].get(SpanAttributes.RECORD_ID),
        )
        # Test multiple record ids.
        res = connector.get_events(
            app_id=tru_app.app_id,
            record_ids=[kojikun_record_id, sachiboy_record_id],
        )
        self.assertEqual(len(res), 2)
        self.assertEqual(
            {kojikun_record_id, sachiboy_record_id},
            set(
                res["record_attributes"].apply(
                    lambda curr: curr.get(SpanAttributes.RECORD_ID)
                )
            ),
        )
