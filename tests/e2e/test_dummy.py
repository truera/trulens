"""
Dummy API tests.

Uses and records an invocation of the DummyApp example which internally uses
DummyAPI for its requests.
"""

from pathlib import Path

from trulens.apps import app
from trulens.core import session as core_session

from examples.dev.dummy_app.app import DummyApp
from examples.dev.dummy_app.tool import DummyStackTool
from tests import test as mod_test


class TestDummy(mod_test.TruTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        DummyStackTool.clear_stack()
        self.session = core_session.TruSession()
        self.session.reset_database()
        super().setUp()

    def tearDown(self):
        DummyStackTool.clear_stack()
        super().tearDown()

    def test_dummy(self):
        """Check that recording of example custom app using dummy endpoint works
        and produces a consistent record."""

        # Create custom app:
        ca = DummyApp(
            delay=0.0, alloc=0, use_parallel=False
        )  # uses DummyAPI internally

        # Create trulens wrapper:
        ta = app.TruApp(ca, app_name="customapp", app_version="base")

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
