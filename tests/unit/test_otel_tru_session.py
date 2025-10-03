from typing import List

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase


class TestOtelTruSession(OtelTestCase):
    def test_get_records_and_feedback(self):
        # Create app.
        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "Kojikun"

        app = _TestApp()
        tru_app = TruApp(
            app, app_name="test_get_records_and_feedback", app_version="v1"
        )
        # Record multiple records.
        with tru_app as recording:
            app.query("Who is the best baby?")
            app.query("Who is the cutest baby?")
        record_ids = [rec.record_id for rec in recording.records]
        # Get records and feedback.
        tru_session = TruSession()
        tru_session.force_flush()

        def num_records(record_ids: List[str]) -> int:
            return len(
                tru_session.get_records_and_feedback(record_ids=record_ids)[0]
            )

        self.assertEqual(2, len(record_ids))
        self.assertEqual(2, num_records(record_ids))
        self.assertEqual(1, num_records([record_ids[0]]))
        self.assertEqual(1, num_records([record_ids[1]]))
        self.assertEqual(0, num_records([]))
