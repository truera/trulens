"""Tests of various functionalities of the `TruSession` class."""

from concurrent.futures import Future as FutureClass
from concurrent.futures import wait
from datetime import datetime
from pathlib import Path
from subprocess import check_call
import time
from unittest import TestCase
import uuid

import pytest
from trulens.apps import basic as basic_app
from trulens.apps import custom as custom_app
from trulens.apps import virtual as virtual_app
from trulens.core import session as core_session
from trulens.core.feedback import feedback as core_feedback
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import keys as key_utils
from trulens.providers.huggingface import provider as huggingface_provider

from tests.unit import feedbacks as feedback_tests


class TestTru(TestCase):
    @staticmethod
    def setUpClass():
        pass

    def setUp(self):
        key_utils.check_keys(
            "OPENAI_API_KEY",
            "HUGGINGFACE_API_KEY",
            # "PINECONE_API_KEY", # don't seem to be needed
            # "PINECONE_ENV",
        )

    def test_init(self):
        """Test Tru class constructor.

        This involves just database-related specifications right now.
        """

        # Try all combinations of arguments to Tru constructor.
        test_args = dict()
        test_args["database_url"] = [None, "sqlite:///some_filename.db"]
        test_args["database_redact_keys"] = [None, True, False]

        for url in test_args["database_url"]:
            for redact in test_args["database_redact_keys"]:
                with self.subTest(url=url, redact=redact):
                    args = dict()
                    if url is not None:
                        args["database_url"] = url
                    if redact is not None:
                        args["database_redact_keys"] = redact

                    try:
                        session = core_session.TruSession(**args)
                    finally:
                        if session is not None:
                            session.delete_singleton()

                    if session is None:
                        continue

                    # Do some db operations to the expected files get created.
                    session.reset_database()

                    # Check that the expected files were created.
                    if url is not None:
                        self.assertTrue(Path("some_filename.db").exists())
                    else:
                        self.assertTrue(Path("default.sqlite").exists())

                # Need to delete singleton after test as otherwise we
                # cannot change the arguments in next test.

    def _create_custom(self):
        from examples.dev.dummy_app.app import DummyApp

        return DummyApp()

    def _create_basic(self):
        def custom_application(prompt: str) -> str:
            return "a response"

        return custom_application

    def _create_chain(self):
        # Note that while langchain is required, openai is not so tests using
        # this app are optional.
        from langchain.prompts import PromptTemplate
        from langchain.schema import StrOutputParser
        from langchain_openai import OpenAI

        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = OpenAI(temperature=0.0, streaming=False, cache=False)
        chain = prompt | llm | StrOutputParser()
        return chain

    def _create_feedback_functions(self):
        provider = huggingface_provider.Dummy(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=1024,  # how much fake data to allocate during requests
        )

        f_dummy1 = core_feedback.Feedback(
            provider.language_match, name="language match"
        ).on_input_output()

        f_dummy2 = core_feedback.Feedback(
            provider.positive_sentiment, name="output sentiment"
        ).on_output()

        f_dummy3 = core_feedback.Feedback(
            provider.positive_sentiment, name="input sentiment"
        ).on_input()

        return [f_dummy1, f_dummy2, f_dummy3]

    def _create_llama(self):
        # Starter example of
        # https://docs.llamaindex.ai/en/latest/getting_started/starter_example.html

        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import VectorStoreIndex

        path = Path("data")

        if not path.exists():
            try:
                check_call([
                    "wget",
                    "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
                    "-P",
                    "data/",
                ])
            except Exception as e:
                raise RuntimeError(
                    "Could not download required text data."
                ) from e

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        return query_engine

    def test_required_constructors(self):
        """
        Test the capitalized methods of Tru class that are aliases for various
        app types. This test includes only ones that do not require optional
        packages.
        """
        core_session.TruSession()

        with self.subTest(type="TruBasicApp"):
            app = self._create_basic()

            with self.subTest(argname=None):
                basic_app.TruBasicApp(app)

            with self.subTest(argname="text_to_text"):
                basic_app.TruBasicApp(text_to_text=app)

            # Not specifying callable should be an error.
            with self.assertRaises(Exception):
                basic_app.TruBasicApp()
            with self.assertRaises(Exception):
                basic_app.TruBasicApp(None)

            # Specifying custom basic app using any of these other argument
            # names should be an error.
            wrong_args = ["app", "chain", "engine"]

            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        basic_app.TruBasicApp(**{arg: app})

        with self.subTest(type="TruCustomApp"):
            app = self._create_custom()

            custom_app.TruCustomApp(app)
            custom_app.TruCustomApp(app=app)

            # Not specifying callable should be an error.
            with self.assertRaises(Exception):
                custom_app.TruCustomApp()
            with self.assertRaises(Exception):
                custom_app.TruCustomApp(None)

            # Specifying custom app using any of these other argument names
            # should be an error.
            wrong_args = ["chain", "engine", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        custom_app.TruCustomApp(**{arg: app})

        with self.subTest(type="TruVirtual"):
            virtual_app.TruVirtual(None)

    @pytest.mark.optional
    def test_langchain_constructors(self):
        """
        Test TruChain class that require optional packages.
        """
        from trulens.apps.langchain.tru_chain import TruChain

        with self.subTest(type="TruChain"):
            app = self._create_chain()

            with self.subTest(argname=None):
                TruChain(app)

            with self.subTest(argname="app"):
                TruChain(app=app)

            # Not specifying chain should be an error.
            with self.assertRaises(Exception):
                TruChain()
            with self.assertRaises(Exception):
                TruChain(None)

            # Specifying the chain using any of these other argument names
            # should be an error.
            wrong_args = ["engine", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        TruChain(**{arg: app})

    @pytest.mark.optional
    def test_llamaindex_constructors(self):
        """
        Test TruLlama class that require optional packages.
        """
        from trulens.apps.llamaindex.tru_llama import TruLlama

        with self.subTest(type="TruLlama"):
            app = self._create_llama()

            TruLlama(app)

            TruLlama(app=app)

            # Not specifying an engine should be an error.
            with self.assertRaises(Exception):
                TruLlama()

            with self.assertRaises(Exception):
                TruLlama(None)

            # Specifying engine using any of these other argument names
            # should be an error.
            wrong_args = ["chain", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        TruLlama(**{arg: app})

    def test_run_feedback_functions_wait(self):
        """
        Test run_feedback_functions in wait mode. This mode blocks until results
        are ready.
        """

        app = self._create_custom()

        feedbacks = self._create_feedback_functions()

        expected_feedback_names = {f.name for f in feedbacks}

        session = core_session.TruSession()

        tru_app = custom_app.TruCustomApp(app)

        with tru_app as recording:
            app.respond_to_query("hello")

        record = recording.get()

        feedback_results = list(
            session.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=True,
            )
        )

        # Check we get the right number of results.
        self.assertEqual(len(feedback_results), len(feedbacks))

        # Check that the results are for the feedbacks we submitted.
        self.assertEqual(
            set(expected_feedback_names),
            set(res.name for res in feedback_results),
            "feedback result names do not match requested feedback names",
        )

        # Check that the structure of returned tuples is correct.
        for result in feedback_results:
            self.assertIsInstance(result, feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        session.add_feedbacks(feedback_results)

        # Check that results were added to db.
        _, returned_feedback_names = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )

        # Check we got the right feedback names from db.
        self.assertEqual(expected_feedback_names, set(returned_feedback_names))

    def test_run_feedback_functions_nowait(self):
        """
        Test run_feedback_functions in non-blocking mode. This mode returns
        futures instead of results.
        """

        app = self._create_custom()

        feedbacks = self._create_feedback_functions()
        expected_feedback_names = {f.name for f in feedbacks}

        session = core_session.TruSession()

        tru_app = custom_app.TruCustomApp(app)

        with tru_app as recording:
            app.respond_to_query("hello")

        record = recording.get()

        start_time = datetime.now()

        future_feedback_results = list(
            session.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=False,
            )
        )

        end_time = datetime.now()

        # Should return quickly.
        self.assertLess(
            (end_time - start_time).total_seconds(),
            2.0,  # TODO: get it to return faster
            "Non-blocking run_feedback_functions did not return fast enough.",
        )

        # Check we get the right number of results.
        self.assertEqual(len(future_feedback_results), len(feedbacks))

        feedback_results = []

        # Check that the structure of returned tuples is correct.
        for future_result in future_feedback_results:
            self.assertIsInstance(future_result, FutureClass)

            wait([future_result])

            result = future_result.result()
            self.assertIsInstance(result, feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

            feedback_results.append(result)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        session.add_feedbacks(feedback_results)

        # Check that results were added to db.
        _, returned_feedback_names = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )

        # Check we got the right feedback names.
        self.assertEqual(expected_feedback_names, set(returned_feedback_names))

    def test_reset_database(self):
        # TODO
        pass

    def test_add_record(self):
        # TODO
        pass

    # def test_add_app(self):
    #     app_name = "test_app"
    #     app_definition = mod_app_schema.AppDefinition(app_name=app_name, model_dump_json="{}")
    #     app_id = app_definition.app_id
    #     session = mod_session.TruSession()
    #     # Action: Add the app to the database
    #     added_app_id = session.add_app(app_definition)

    #     # Assert: Verify the app was added successfully
    #     self.assertEqual(app_id, added_app_id)
    #     retrieved_app = session.get_app(app_id)
    #     self.assertIsNotNone(retrieved_app)
    #     self.assertEqual(retrieved_app['app_id'], app_id)

    # def test_delete_app(self):
    #     # Setup: Add an app to the database
    #     app_name = "test_app"
    #     app_definition = mod_app_schema.AppDefinition(app_name=app_name, model_dump_json="{}")
    #     app_id = app_definition.app_id
    #     session = mod_session.TruSession()
    #     session.add_app(app_definition)

    #     # Action: Delete the app
    #     session.delete_app(app_id)

    #     # Assert: Verify the app is deleted
    #     retrieved_app = session.get_app(app_id)
    #     self.assertIsNone(retrieved_app)

    def test_add_feedback(self):
        # TODO
        pass

    def test_add_feedbacks(self):
        # TODO: move testing from test_run_feedback_functions_wait and
        # test_run_feedback_functions_nowait.
        pass

    def test_get_records_and_feedback(self):
        # Also tested in test_run_feedback_functions_wait and
        # test_run_feedback_functions_nowait.
        # TODO
        pass

    def test_get_leaderboard(self):
        # TODO
        pass

    def test_start_evaluator(self):
        # TODO
        pass

    def test_start_evaluator_with_blocking(self):
        session = core_session.TruSession()
        f = core_feedback.Feedback(
            feedback_tests.custom_feedback_function
        ).on_default()
        app_name = f"test_start_evaluator_with_blocking_{str(uuid.uuid4())}"
        tru_app = basic_app.TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f],
            feedback_mode=feedback_schema.FeedbackMode.DEFERRED,
            app_name=app_name,
        )
        with tru_app:
            tru_app.main_call("test_deferred_mode")
        time.sleep(2)
        session.start_evaluator(return_when_done=True)
        if session._evaluator_proc is not None:
            # We should never get here since the variable isn't supposed to be set.
            raise ValueError("The evaluator is still running!")
        records_and_feedback = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["custom_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["custom_feedback_function"].iloc[0],
            0.1,
        )

    def test_stop_evaluator(self):
        # TODO
        pass

    def test_stop_dashboard(self):
        # TODO
        pass

    def test_run_dashboard(self):
        pass
