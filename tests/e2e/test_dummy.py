"""
Dummy API tests.

Uses and records an invocation of the DummyApp example which internally uses
DummyAPI for its requests.
"""

from unittest import main

from trulens.core.app.custom import TruCustomApp
from trulens.core.tru import Tru

from examples.dev.dummy_app.app import DummyApp
from tests.test import JSONTestCase


class TestDummy(JSONTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        self.tru = Tru()
        self.tru.reset_database()

    def test_dummy(self):
        """Check that recording of example custom app using dummy endpoint works
        and produces a consistent record."""

        # Create custom app:
        ca = DummyApp(
            delay=0.0, alloc=0, use_parallel=True
        )  # uses DummyAPI internally

        # Create trulens wrapper:
        ta = TruCustomApp(ca, app_id="customapp")

        with ta as recorder:
            ca.respond_to_query("hello")

        rec = recorder.get()

        self.assertGoldenJSONEqual(
            actual=rec.model_dump(),
            golden_filename="dummy.json",
            skips=set([
                "record_id",
                "start_time",
                "end_time",
                "ts",
                "pid",
                "tid",
                "call_id",
                "id",
            ]),
            unordereds=set(["calls"]),
        )


if __name__ == "__main__":
    main()
