"""
Tests for Feedback class. 
"""

from unittest import main
from unittest import TestCase

# Get the "globally importable" feedback implementations.
from tests.unit.feedbacks import custom_feedback_function
from tests.unit.feedbacks import CustomClassNoArgs
from tests.unit.feedbacks import CustomClassWithArgs
from tests.unit.feedbacks import CustomProvider
from tests.unit.feedbacks import make_nonglobal_feedbacks

from trulens_eval import Feedback
from trulens_eval import Tru
from trulens_eval import TruCustomApp
from trulens_eval.keys import check_keys
from trulens_eval.schema import FeedbackMode
from trulens_eval.tru_basic_app import TruBasicApp
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.utils.json import jsonify


class TestFeedbackConstructors(TestCase):

    def setUp(self):
        check_keys(
            "OPENAI_API_KEY", "HUGGINGFACE_API_KEY", "PINECONE_API_KEY",
            "PINECONE_ENV"
        )

        self.app = TruBasicApp(text_to_text=lambda t: f"returning {t}")
        _, self.record = self.app.with_record(self.app.app, t="hello")

    def test_global_feedback_functions(self):
        # NOTE: currently static methods and class methods are not supported

        for imp, target in [
            (custom_feedback_function, 0.1),
                # (CustomProvider.static_method, 0.2),
                # (CustomProvider.class_method, 0.3),
            (CustomProvider(attr=0.37).method, 0.4 + 0.37),
                # (CustomClassNoArgs.static_method, 0.5),
                # (CustomClassNoArgs.class_method, 0.6),
            (CustomClassNoArgs().method, 0.7),
                # (CustomClassWithArgs.static_method, 0.8),
                # (CustomClassWithArgs.class_method, 0.9),
                # (CustomClassWithArgs(attr=0.37).method, 1.0 + 0.73)
        ]:

            with self.subTest(imp=imp, taget=target):
                f = Feedback(imp).on_default()

                # Run the feedback function.
                res = f.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

                # Serialize and deserialize the feedback function.
                fs = f.model_dump()

                fds = Feedback.model_validate(fs)

                # Run it again.
                res = fds.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

    def test_global_unsupported(self):
        # Each of these should fail when trying to serialize/deserialize.

        for imp, target in [
                # (custom_feedback_function, 0.1),
                # (CustomProvider.static_method, 0.2), # TODO
            (CustomProvider.class_method, 0.3),
                # (CustomProvider(attr=0.37).method, 0.4 + 0.37),
                # (CustomClassNoArgs.static_method, 0.5), # TODO
            (CustomClassNoArgs.class_method, 0.6),
                # (CustomClassNoArgs().method, 0.7),
                # (CustomClassWithArgs.static_method, 0.8), # TODO
            (CustomClassWithArgs.class_method, 0.9),
            (CustomClassWithArgs(attr=0.37).method, 1.0 + 0.73)
        ]:

            with self.subTest(imp=imp, taget=target):
                f = Feedback(imp).on_default()
                with self.assertRaises(Exception):
                    Feedback.model_validate(f.model_dump())

    def test_nonglobal_feedback_functions(self):
        # Set up the same feedback functions as in feedback.py but locally here.
        # This makes them non-globally-importable.

        NG = make_nonglobal_feedbacks()

        for imp, target in [
            (NG.NGcustom_feedback_function, 0.1),
                # (NG.CustomProvider.static_method, 0.2),
                # (NG.CustomProvider.class_method, 0.3),
            (NG.NGCustomProvider(attr=0.37).method, 0.4 + 0.37),
                # (NG.CustomClassNoArgs.static_method, 0.5),
                # (NG.CustomClassNoArgs.class_method, 0.6),
            (NG.NGCustomClassNoArgs().method, 0.7),
                # (NG.CustomClassWithArgs.static_method, 0.8),
                # (NG.CustomClassWithArgs.class_method, 0.9),
                # (NG.CustomClassWithArgs(attr=0.37).method, 1.0 + 0.73)
        ]:

            with self.subTest(imp=imp, taget=target):
                f = Feedback(imp).on_default()

                # Run the feedback function.
                res = f.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

                # Serialize and deserialize the feedback function.
                fs = f.model_dump()

                # This should fail:
                with self.assertRaises(Exception):
                    fds = Feedback.model_validate(fs)

                # OK to use with App as long as not deferred mode:
                TruBasicApp(
                    text_to_text=lambda t: f"returning {t}",
                    feedbacks=[f],
                    feedback_mode=FeedbackMode.WITH_APP
                )

                # OK to use with App as long as not deferred mode:
                TruBasicApp(
                    text_to_text=lambda t: f"returning {t}",
                    feedbacks=[f],
                    feedback_mode=FeedbackMode.WITH_APP_THREAD
                )

                # Trying these feedbacks with an app with deferred mode should
                # fail at app construction:
                with self.assertRaises(Exception):
                    TruBasicApp(
                        text_to_text=lambda t: f"returning {t}",
                        feedbacks=[f],
                        feedback_mode=FeedbackMode.DEFERRED
                    )


if __name__ == '__main__':
    main()
