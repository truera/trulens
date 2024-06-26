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

from trulens_eval import Tru
from trulens_eval.feedback.provider.dummy import DummyProvider
from trulens_eval.tru_custom_app import TruCustomApp

pp = PrettyPrinter()


class TestDummy(TestCase):
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
            alloc = 0,
            delay = 0.0
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

        actual = rec.model_dump()

        golden_path = (Path(__file__).parent / "golden" / "dummy.json").resolve()

        if self.write_golden:
            with golden_path.open("w") as f:
                json.dump(actual, f)

            self.fail("Golden file written.")

        else:
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden file {golden_path} not found.")
           
            with golden_path.open("r") as f:
                expected = json.load(f)

            # NEED JSON DIFF HERE
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    main()
