"""
Tests for TruBasicApp.
"""

import unittest
from unittest import main

from tests.unit.test import JSONTestCase

from trulens_eval import Tru
from trulens_eval.keys import check_keys
from trulens_eval import TruBasicApp


check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")


class TestTruBasicApp(JSONTestCase):

    def setUp(self):
        def custom_application(prompt: str) -> str:
            return "a response"

        # Temporary before db migration gets fixed.
        Tru().migrate_database()

        self.basic_app = custom_application

        self.tru_basic_app = TruBasicApp(self.basic_app, app_id="Custom Application v1")

    
    def test_no_fail(self):
        # Most naive test to make sure the basic app runs at all.

        msg = "What is the phone number for HR?"

        res1 = self.basic_app(msg)
        res2, rec2 = self.tru_basic_app.call_with_record(msg)

        self.assertJSONEqual(res1, res2)
        self.assertIsNotNone(rec2)
    
if __name__ == '__main__':
    main()
