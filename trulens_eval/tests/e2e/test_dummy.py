"""
# Tests endpoints.

These tests make use of potentially non-free apis and require
various secrets configured. See `setUp` below.
"""

import json
import os
from pathlib import Path
from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase

from examples.expositional.end2end_apps.custom_app.custom_app import CustomApp
from tests.unit.test import JSONTestCase

from trulens_eval import Tru
from trulens_eval.feedback.provider.dummy import DummyProvider
from trulens_eval.tru_custom_app import TruCustomApp

pp = PrettyPrinter()


class TestDummy(JSONTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        self.tru = Tru()
        self.tru.reset_database()

        self.write_golden: bool = bool(os.environ.get("WRITE_GOLDEN", ""))

    def test_dummy(self):
        """Check that recording of example custom app using dummy endpoint works
        and produces a consistent record."""

        d = DummyProvider(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=0,
            delay=0.0
        )

        # Create custom app:
        ca = CustomApp(delay=0.0, alloc=0)

        # Create trulens wrapper:
        ta = TruCustomApp(
            ca,
            app_id="customapp",
        )

        with ta as recorder:
            res = ca.respond_to_query(f"hello")

        rec = recorder.get()

        self.assertGoldenJSONEqual(
            actual=rec.model_dump(),
            golden_filename="dummy.json",
            skips=set(
                [
                    "record_id", 'start_time', 'end_time', 'ts', 'pid', 'tid',
                    'call_id', 'id'
                ]
            )
        )


if __name__ == '__main__':
    main()
