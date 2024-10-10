"""Tests serialization of various components into JSON.

These tests make sure that:

1. Serialization does not fail.

2. Produced json matches the golden expected results already collected as part
   of this test.

To refresh the golden files, set the environment variable `WRITE_GOLDEN` to
anything that evaluates to true. This will overwrite the golden files with the
actual results produced by the tests. Only do this if changes to serialization
are expected.
"""

from pathlib import Path
from unittest import main

from trulens.apps import custom as custom_app
from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import threading as threading_utils
from trulens.feedback.dummy import provider as dummy_provider
from trulens.providers.huggingface import provider as huggingface_provider

from examples.dev.dummy_app.app import DummyApp
from tests import test as mod_test

_GOLDEN_PATH = Path("tests") / "e2e" / "golden"


class TestSerial(mod_test.TruTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        pass

    def tearDown(self):
        # Need to shutdown threading pools as otherwise the thread cleanup
        # checks will fail.
        threading_utils.TP().shutdown()

        super().tearDown()

    def test_app_serial(self):
        """Check that the custom app and products are serialized consistently."""

        ca = DummyApp(delay=0.0, alloc=0, use_parallel=False)

        d = dummy_provider.DummyProvider(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=0,
            delay=0.0,
        )

        d_hugs = huggingface_provider.Dummy(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=0,
            delay=0.0,
        )

        feedback_language_match = core_feedback.Feedback(
            d_hugs.language_match
        ).on_input_output()
        feedback_context_relevance = core_feedback.Feedback(
            d.context_relevance
        ).on_input_output()

        ta = custom_app.TruCustomApp(
            ca,
            app_name="customapp",
            feedbacks=[feedback_language_match, feedback_context_relevance],
        )

        with self.subTest(step="app serialization"):
            self.assertGoldenJSONEqual(
                actual=ta.model_dump(),
                golden_path=_GOLDEN_PATH / "customapp.json",
                skips=set([
                    "app_id",
                    "feedback_definitions",  # contains ids
                ]),
            )

        with ta as recorder:
            res = ca.respond_to_query("hello")

        with self.subTest(step="app result serialization"):
            self.assertGoldenJSONEqual(
                actual=res,
                golden_path=_GOLDEN_PATH / "customapp_result.json",
            )

        record = recorder.get()

        with self.subTest(step="record serialization"):
            self.assertGoldenJSONEqual(
                actual=record.model_dump(),
                golden_path=_GOLDEN_PATH / "customapp_record.json",
                skips=set([
                    "app_id",
                    "end_time",
                    "start_time",
                    "record_id",
                    "pid",
                    "tid",
                    "id",
                    "ts",
                    "call_id",
                ]),
            )

        feedbacks = record.wait_for_feedback_results()

        for fdef, fres in feedbacks.items():
            name = fdef.name

            with self.subTest(step=f"feedback definition {name} serialization"):
                self.assertGoldenJSONEqual(
                    actual=fdef.model_dump(),
                    golden_path=_GOLDEN_PATH / f"customapp_{name}.def.json",
                    skips=set([
                        "feedback_definition_id",
                        "id",
                    ]),
                )

            with self.subTest(step=f"feedback result {name} serialization"):
                self.assertGoldenJSONEqual(
                    actual=fres.model_dump(),
                    golden_path=_GOLDEN_PATH / f"customapp_{name}.result.json",
                    skips=set([
                        "feedback_definition_id",
                        "id",
                        "last_ts",
                        "record_id",
                        "feedback_result_id",
                        # Skip these if non-determinism becomes a problem:
                        "result",
                        "LABEL_0",
                        "LABEL_1",
                        "LABEL_2",
                        "ret",
                    ]),
                )


if __name__ == "__main__":
    main()
