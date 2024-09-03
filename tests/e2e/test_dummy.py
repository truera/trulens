"""
Dummy API tests.

Uses and records an invocation of the DummyApp example which internally uses
DummyAPI for its requests.
"""

from unittest import main

from trulens.apps.custom import TruCustomApp
from trulens.core import TruSession

from examples.dev.dummy_app.app import DummyApp
from tests.test import JSONTestCase


class TestDummy(JSONTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        self.session = TruSession()
        self.session.reset_database()

    def test_dummy(self):
        """Check that recording of example custom app using dummy endpoint works
        and produces a consistent record."""

        # Create custom app:
        ca = DummyApp(
            delay=0.0, alloc=0, use_parallel=True
        )  # uses DummyAPI internally

        # Create trulens wrapper:
        ta = TruCustomApp(ca, app_name="customapp", app_version="base")

        with ta as recorder:
            ca.respond_to_query("hello")

        rec = recorder.get()

        self.assertGoldenJSONEqual(
            actual=rec.model_dump(),
            golden_filename="tests/e2e/golden/dummy.json",
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
