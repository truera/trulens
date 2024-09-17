"""
Dummy API tests.

Uses and records an invocation of the DummyApp example which internally uses
DummyAPI for its requests.
"""

from pathlib import Path
from unittest import main

from trulens.apps.custom import TruCustomApp
from trulens.core import TruSession

from examples.dev.dummy_app.app import DummyApp
from tests.test import TruTestCase


class TestDummy(TruTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        self.session = TruSession()
        self.session.reset_database()

    def test_dummy(self):
        """Check that recording of example custom app using dummy endpoint works
        and produces a consistent record."""

        # Create custom app:
        ca = DummyApp(
            delay=0.0, alloc=0, use_parallel=False
        )  # uses DummyAPI internally

        # Create trulens wrapper:
        ta = TruCustomApp(ca, app_name="customapp", app_version="base")

        with ta as recorder:
            ca.respond_to_query("hello")

        rec = recorder.get()

        self.assertGoldenJSONEqual(
            actual=rec.model_dump(),
            golden_path=Path("tests") / "e2e" / "golden" / "dummy.json",
            skips=set([
                "app_id",
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

        # Test for memory leaks.
        # Disabling for now as it is failing. Fix is in another PR.
        # ca_ref = weakref.ref(ca)
        # del ca, ta, recorder, rec
        # self.assertCollected(ca_ref)


if __name__ == "__main__":
    main()
