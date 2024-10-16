"""Tests for TruBasicApp."""

from unittest import main

from trulens.apps import basic as basic_app
from trulens.core import session as core_session
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import keys as key_utils

from tests import test as mod_test

key_utils.check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")


class TestTruBasicApp(mod_test.TruTestCase):
    def setUp(self):
        def custom_application(prompt: str) -> str:
            return "a response"

        self.session = core_session.TruSession()

        # Temporary before db migration gets fixed.
        self.session.migrate_database()

        # Reset database here.
        self.session.reset_database()

        self.basic_app = custom_application

        self.tru_basic_app_recorder = basic_app.TruBasicApp(
            self.basic_app,
            app_name="Custom Application",
            app_version="v1",
            feedback_mode=feedback_schema.FeedbackMode.WITH_APP,
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
