"""
# Tests serialization of various components.
"""

from pprint import PrettyPrinter
from unittest import main

from examples.expositional.end2end_apps.custom_app.custom_app import CustomApp
from tests.unit.test import JSONTestCase

from trulens_eval import Feedback
from trulens_eval.feedback.provider.dummy import DummyProvider
from trulens_eval.feedback.provider.hugs import Dummy
from trulens_eval.tru_custom_app import TruCustomApp

pp = PrettyPrinter()


class TestSerial(JSONTestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        pass

    def test_app_serial(self):
        """Check that the custom app and products are serialized consistently."""

        ca = CustomApp(delay=0.0, alloc=0)

        d = DummyProvider(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=0,
            delay=0.0
        )

        d_hugs = Dummy(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=0,
            delay=0.0
        )

        feedback_language_match = Feedback(d_hugs.language_match
                                          ).on_input_output()
        feedback_context_relevance = Feedback(d.context_relevance
                                             ).on_input_output()

        ta = TruCustomApp(
            ca,
            app_id="customapp",
            feedbacks=[feedback_language_match, feedback_context_relevance]
        )

        with self.subTest("app serialization"):
            self.assertGoldenJSONEqual(
                actual=ta.model_dump(),
                golden_filename="customapp.json",
                skips=set([])
            )

        with ta as recorder:
            res = ca.respond_to_query(f"hello")

        with self.subTest("app result serialization"):
            self.assertGoldenJSONEqual(
                actual=res,
                golden_filename="customapp_result.json",
                skips=set([])
            )

        record = recorder.get()

        with self.subTest("record serialization"):
            self.assertGoldenJSONEqual(
                actual=record.model_dump(),
                golden_filename="customapp_record.json",
                skips=set(
                    [
                        'end_time', 'start_time', 'record_id', 'pid', 'tid',
                        'id', 'ts', 'call_id'
                    ]
                )
            )

        feedbacks = record.wait_for_feedback_results()
        for fdef, fres in feedbacks.items():
            name = fdef.name
            with self.subTest(f"feedback definition {name} serialization"):
                self.assertGoldenJSONEqual(
                    actual=fdef.model_dump(),
                    golden_filename=f"customapp_{name}.def.json",
                    skips=set(['feedback_definition_id', 'id'])
                )
            with self.subTest(f"feedback result {name} serialization"):
                self.assertGoldenJSONEqual(
                    actual=fres.model_dump(),
                    golden_filename=f"customapp_{name}.result.json",
                    skips=set(['feedback_definition_id', 'id', 'last_ts', 'record_id', 'feedback_result_id', 'result', 'LABEL_0', 'LABEL_1', 'LABEL_2', 'ret'])
                    # Having trouble controlling result determinism for this test.
                )


if __name__ == '__main__':
    main()
