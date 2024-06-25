"""
# Tests endpoints.

These tests make use of potentially non-free apis and require
various secrets configured. See `setUp` below.
"""

from concurrent.futures import as_completed
import os
from pprint import PrettyPrinter
from time import sleep
from unittest import main
from unittest import TestCase

from examples.expositional.end2end_apps.custom_app.custom_app import CustomApp
from tests.unit.test import optional_test
from tqdm.auto import tqdm

from trulens_eval import Feedback
from trulens_eval import Tru
from trulens_eval.feedback.provider.dummy import DummyProvider
from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.keys import check_keys
from trulens_eval.schema.feedback import FeedbackMode
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.threading import TP

pp = PrettyPrinter()


class TestDummy(TestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        self.tru = Tru()
        self.tru.reset_database()

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

        print(rec.model_dump())

if __name__ == '__main__':
    main()
