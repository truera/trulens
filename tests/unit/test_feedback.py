"""Tests for Feedback class."""

import time
from unittest import TestCase

import numpy as np
import pytest
from trulens.apps import basic as basic_app
from trulens.apps.langchain import TruChain
from trulens.core import Feedback
from trulens.core import TruSession
from trulens.core.feedback import feedback as core_feedback
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import select as select_schema
from trulens.providers.langchain import Langchain

# Get the "globally importable" feedback implementations.
from tests.unit import feedbacks as test_feedbacks
import tests.unit.test_otel_tru_chain


class TestFeedbackEval(TestCase):
    """Tests for feedback function evaluation."""

    def test_skipeval(self) -> None:
        """Test the SkipEval capability."""

        f = core_feedback.Feedback(imp=test_feedbacks.skip_if_odd).on(
            val=select_schema.Select.RecordCalls.somemethod.args.num[:]
        )

        # Create source data that looks like real source data for a record
        # collected from a real app. Store some integers in a place that
        # corresponds to app call to `somemethod`, keyword argument `num`.
        source_data = {
            "__record__": {
                "app": {"somemethod": {"args": {"num": [1, 2, 3, 4, 5, 6]}}}
            }
        }

        res = f.run(source_data=source_data)

        self.assertNotAlmostEqual((1 + 2 + 3 + 4 + 5 + 6) / 6, (2 + 4 + 6) / 3)
        # Make sure that the wrong behavior is not accidentally equal to the
        # correct one.

        self.assertIsInstance(res.result, float)

        self.assertAlmostEqual(res.result, (2 + 4 + 6) / 3)
        # Odds should have been skipped.

        self.assertEqual(res.status, feedback_schema.FeedbackResultStatus.DONE)
        # Status should be DONE.

    def test_skipeval_all(self) -> None:
        """Test the SkipEval capability for when all evals are skipped"""

        f = core_feedback.Feedback(imp=test_feedbacks.skip_if_odd).on(
            val=select_schema.Select.RecordCalls.somemethod.args.num[:]
        )

        # Create source data that looks like real source data for a record
        # collected from a real app. Store some integers in a place that
        # corresponds to app call to `somemethod`, keyword argument `num`.
        source_data = {
            "__record__": {"app": {"somemethod": {"args": {"num": [1, 3, 5]}}}}
        }

        res = f.run(source_data=source_data)

        self.assertIsInstance(res.result, float)

        self.assertTrue(np.isnan(res.result))

        self.assertEqual(res.status, feedback_schema.FeedbackResultStatus.DONE)
        # But status should be DONE (as opposed to SKIPPED or ERROR)

    @pytest.mark.optional
    def test_same_provider_for_app_and_feedback(self) -> None:
        tru_session = TruSession()
        tru_session.reset_database()
        rag_chain = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        llm = rag_chain.middle[
            1
        ]  # Bit of a hack, but this is the LLM for the chain.
        provider = Langchain(chain=llm)
        f_answer_relevance = Feedback(
            provider.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()
        tru_recorder = TruChain(
            rag_chain,
            app_name="ChatApplication",
            app_version="Chain1",
            feedbacks=[f_answer_relevance],
        )
        with tru_recorder:
            rag_chain.invoke("What is Task Decomposition?")
            time.sleep(1)
        res = tru_session.get_records_and_feedback()[0]
        self.assertEqual(len(res), 1)


class TestFeedbackConstructors(TestCase):
    """Test for feedback function serialization/deserialization."""

    def setUp(self) -> None:
        self.app = basic_app.TruBasicApp(
            text_to_text=lambda t: f"returning {t}"
        )
        _, self.record = self.app.with_record(self.app.app, t="hello")

    def test_global_feedback_functions(self) -> None:
        # NOTE: currently static methods and class methods are not supported

        for imp, target in [
            (test_feedbacks.custom_feedback_function, 0.1),
            # (test_feedbacks.CustomProvider.static_method, 0.2),
            # (test_feedbacks.CustomProvider.class_method, 0.3),
            (test_feedbacks.CustomProvider(attr=0.37).method, 0.4 + 0.37),
            # (test_feedbacks.CustomClassNoArgs.static_method, 0.5),
            # (test_feedbacks.CustomClassNoArgs.class_method, 0.6),
            (test_feedbacks.CustomClassNoArgs().method, 0.7),
            # (test_feedbacks.CustomClassWithArgs.static_method, 0.8),
            # (test_feedbacks.CustomClassWithArgs.class_method, 0.9),
            # (test_feedbacks.CustomClassWithArgs(attr=0.37).method, 1.0 + 0.73)
        ]:
            with self.subTest(imp=imp, target=target):
                f = core_feedback.Feedback(imp).on_default()

                # Run the feedback function.
                res = f.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

                # Serialize and deserialize the feedback function.
                fs = f.model_dump()

                fds = core_feedback.Feedback.model_validate(fs)

                # Run it again.
                res = fds.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

    def test_global_unsupported(self) -> None:
        # Each of these should fail when trying to serialize/deserialize.

        for imp, target in [
            # (test_feedbacks.custom_feedback_function, 0.1),
            # (test_feedbacks.CustomProvider.static_method, 0.2), # TODO
            (test_feedbacks.CustomProvider.class_method, 0.3),
            # (test_feedbacks.CustomProvider(attr=0.37).method, 0.4 + 0.37),
            # (test_feedbacks.CustomClassNoArgs.static_method, 0.5), # TODO
            (test_feedbacks.CustomClassNoArgs.class_method, 0.6),
            # (test_feedbacks.CustomClassNoArgs().method, 0.7),
            # (test_feedbacks.CustomClassWithArgs.static_method, 0.8), # TODO
            (test_feedbacks.CustomClassWithArgs.class_method, 0.9),
            (test_feedbacks.CustomClassWithArgs(attr=0.37).method, 1.0 + 0.73),
        ]:
            with self.subTest(imp=imp, target=target):
                f = core_feedback.Feedback(imp).on_default()
                with self.assertRaises(Exception):
                    core_feedback.Feedback.model_validate(f.model_dump())

    def test_nonglobal_feedback_functions(self) -> None:
        # Set up the same feedback functions as in feedback.py but locally here.
        # This makes them non-globally-importable.

        NG = test_feedbacks.make_nonglobal_feedbacks()

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
            with self.subTest(imp=imp, target=target):
                f = core_feedback.Feedback(imp).on_default()

                # Run the feedback function.
                res = f.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

                # Serialize and deserialize the feedback function.
                fs = f.model_dump()

                # This should fail:
                with self.assertRaises(Exception):
                    core_feedback.Feedback.model_validate(fs)

                # OK to use with App as long as not deferred mode:
                basic_app.TruBasicApp(
                    text_to_text=lambda t: f"returning {t}",
                    feedbacks=[f],
                    feedback_mode=feedback_schema.FeedbackMode.WITH_APP,
                )

                # OK to use with App as long as not deferred mode:
                basic_app.TruBasicApp(
                    text_to_text=lambda t: f"returning {t}",
                    feedbacks=[f],
                    feedback_mode=feedback_schema.FeedbackMode.WITH_APP_THREAD,
                )

                # Trying these feedbacks with an app with deferred mode should
                # fail at app construction:
                with self.assertRaises(Exception):
                    basic_app.TruBasicApp(
                        text_to_text=lambda t: f"returning {t}",
                        feedbacks=[f],
                        feedback_mode=feedback_schema.FeedbackMode.DEFERRED,
                    )
