"""
Tests for TruBasicApp.
"""

from unittest import main

from trulens.apps.basic import TruBasicApp
from trulens.core import TruSession
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.utils.keys import check_keys

from tests.test import TruTestCase

check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")


class TestTruBasicApp(TruTestCase):
    def setUp(self):
        def custom_application(prompt: str) -> str:
            return "a response"

        self.session = TruSession()

        # Temporary before db migration gets fixed.
        self.session.migrate_database()

        # Reset database here.
        self.session.reset_database()

        self.basic_app = custom_application

        self.tru_basic_app_recorder = TruBasicApp(
            self.basic_app,
            app_name="Custom Application",
            app_version="v1",
            feedback_mode=FeedbackMode.WITH_APP,
        )

    def test_no_fail(self):
        # Most naive test to make sure the basic app runs at all.

        msg = "What is the phone number for HR?"

        res1 = self.basic_app(msg)
        with self.tru_basic_app_recorder as recording:
            res2 = self.tru_basic_app_recorder.app(msg)

        rec2 = recording.records[0]

        self.assertJSONEqual(res1, res2)
        self.assertIsNotNone(rec2)

        # Check the database has the record
        records = self.session.get_records_and_feedback()[0]

        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    main()
