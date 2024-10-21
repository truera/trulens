"""
Tests for TruCustomApp.
"""

from unittest import main
import weakref

from trulens.apps import custom as custom_app
from trulens.core import session as core_session

from examples.dev.dummy_app.app import DummyApp
from tests.test import TruTestCase


class TestTruCustomApp(TruTestCase):
    @staticmethod
    def setUpClass():
        core_session.TruSession().reset_database()

    def _create_app(self):
        app = DummyApp()
        recorder = custom_app.TruCustomApp(
            app, app_name="custom_app", app_version="v1"
        )

        return app, recorder

    def setUp(self):
        self.session = core_session.TruSession()

    def tearDown(self):
        super().tearDown()

    def test_with_record(self):
        app, recorder = self._create_app()

        question = "What is the capital of Indonesia?"

        # Normal usage:
        response_normal = app.respond_to_query(query=question)

        # Instrumented usage:
        response_wrapped, record = recorder.with_record(
            app.respond_to_query, query=question, record_metadata="meta1"
        )

        self.assertEqual(response_normal, response_wrapped)

        self.assertIsNotNone(record)

        self.assertEqual(record.meta, "meta1")

        # Check GC.
        app_ref = weakref.ref(app)
        recorder_ref = weakref.ref(recorder)
        del app, recorder
        self.assertCollected(app_ref)
        self.assertCollected(recorder_ref)

    def test_context_manager(self):
        app, recorder = self._create_app()

        question = "What is the capital of Indonesia?"

        # Normal usage:
        response_normal = app.respond_to_query(query=question)

        # Instrumented usage:
        with recorder as recording:
            response_wrapped = app.respond_to_query(query=question)

        self.assertEqual(response_normal, response_wrapped)

        self.assertIsNotNone(recording.get())

        # Check GC.
        app_ref = weakref.ref(app)
        recorder_ref = weakref.ref(recorder)
        del app, recorder
        self.assertCollected(app_ref)
        self.assertCollected(recorder_ref)

    def test_nested_context_manager(self):
        app, recorder = self._create_app()

        question1 = "What is the capital of Indonesia?"
        question2 = "What is the capital of Poland?"

        # Normal usage:
        response_normal1 = app.respond_to_query(query=question1)
        response_normal2 = app.respond_to_query(query=question2)

        # Instrumented usage:
        with recorder as recording1:
            recording1.record_metadata = "meta1"
            response_wrapped1 = app.respond_to_query(query=question1)
            with recorder as recording2:
                recording2.record_metadata = "meta2"
                response_wrapped2 = app.respond_to_query(query=question2)

        self.assertEqual(response_normal1, response_wrapped1)
        self.assertEqual(response_normal2, response_wrapped2)

        self.assertEqual(len(recording1.records), 2)
        self.assertEqual(len(recording2.records), 1)

        # Context managers produce similar but not identical records.
        # Specifically, timestamp and meta differ and therefore record_id
        # differs.
        self.assertJSONEqual(
            recording1[1], recording2[0], skips=["record_id", "ts", "meta"]
        )

        self.assertEqual(recording1[0].meta, "meta1")
        self.assertEqual(recording1[1].meta, "meta1")

        self.assertEqual(recording2[0].meta, "meta2")

        # Check GC.
        app_ref = weakref.ref(app)
        recorder_ref = weakref.ref(recorder)
        del app, recorder
        self.assertCollected(app_ref)
        self.assertCollected(recorder_ref)


if __name__ == "__main__":
    main()
