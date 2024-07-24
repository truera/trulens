"""
Tests for TruCustomApp.
"""

from unittest import main

from trulens.core import Tru
from trulens.core import TruCustomApp

from examples.expositional.end2end_apps.custom_app.custom_app import CustomApp
from tests.unit.test import JSONTestCase


class TestTruCustomApp(JSONTestCase):
    @staticmethod
    def setUpClass():
        Tru().reset_database()

    def setUp(self):
        self.tru = Tru()

        self.ca = CustomApp()
        self.ta_recorder = TruCustomApp(self.ca, app_id="custom_app")

    def test_with_record(self):
        question = "What is the capital of Indonesia?"

        # Normal usage:
        response_normal = self.ca.respond_to_query(question)

        # Instrumented usage:
        response_wrapped, record = self.ta_recorder.with_record(
            self.ca.respond_to_query, input=question, record_metadata="meta1"
        )

        self.assertEqual(response_normal, response_wrapped)

        self.assertIsNotNone(record)

        self.assertEqual(record.meta, "meta1")

    def test_context_manager(self):
        question = "What is the capital of Indonesia?"

        # Normal usage:
        response_normal = self.ca.respond_to_query(question)

        # Instrumented usage:
        with self.ta_recorder as recording:
            response_wrapped = self.ca.respond_to_query(input=question)

        self.assertEqual(response_normal, response_wrapped)

        self.assertIsNotNone(recording.get())

    def test_nested_context_manager(self):
        question1 = "What is the capital of Indonesia?"
        question2 = "What is the capital of Poland?"

        # Normal usage:
        response_normal1 = self.ca.respond_to_query(question1)
        response_normal2 = self.ca.respond_to_query(question2)

        # Instrumented usage:
        with self.ta_recorder as recording1:
            recording1.record_metadata = "meta1"
            response_wrapped1 = self.ca.respond_to_query(input=question1)
            with self.ta_recorder as recording2:
                recording2.record_metadata = "meta2"
                response_wrapped2 = self.ca.respond_to_query(input=question2)

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


if __name__ == "__main__":
    main()
