"""
Tests of various functionalities of the `Tru` class.
"""

from concurrent.futures import Future as FutureClass
from concurrent.futures import wait
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from tests.unit.test import optional_test

from trulens_eval import Feedback

from trulens_eval import Tru
from trulens_eval import TruCustomApp
from trulens_eval.feedback.provider.hugs import Dummy
from trulens_eval.keys import check_keys
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.schema import feedback as mod_feedback_schema

class TestTru(TestCase):

    @staticmethod
    def setUpClass():
        pass

    def setUp(self):
        check_keys(
            "OPENAI_API_KEY", "HUGGINGFACE_API_KEY", "PINECONE_API_KEY",
            "PINECONE_ENV"
        )

    def test_init(self):
        """
        Test Tru class constructor. This involves just database-related
        specifications right now.
        """

        # Try all combinations of arguments to Tru constructor.
        test_args = dict()
        test_args['database_url'] = [None, "sqlite:///default_url.db"]
        test_args['database_file'] = [None, "default_file.db"]
        test_args['database_redact_keys'] = [None, True, False]

        tru = None

        for url in test_args['database_url']:
            for file in test_args['database_file']:
                for redact in test_args['database_redact_keys']:
                    with self.subTest(url=url, file=file, redact=redact):
                        args = dict()
                        if url is not None:
                            args['database_url'] = url
                        if file is not None:
                            args['database_file'] = file
                        if redact is not None:
                            args['database_redact_keys'] = redact

                        if url is not None and file is not None:
                            # Specifying both url and file should throw exception.
                            with self.assertRaises(Exception):
                                tru = Tru(**args)

                            if tru is not None:
                                tru.delete_singleton()

                        else:
                            try:
                                tru = Tru(**args)
                            finally:
                                if tru is not None:
                                    tru.delete_singleton()

                            if tru is None:
                                continue

                            # Do some db operations to the expected files get created.
                            tru.reset_database()

                            # Check that the expected files were created.
                            if url is not None:
                                self.assertTrue(Path("default_url.db").exists())
                            elif file is not None:
                                self.assertTrue(
                                    Path("default_file.db").exists()
                                )
                            else:
                                self.assertTrue(Path("default.sqlite").exists())

                        # Need to delete singleton after test as otherwise we
                        # cannot change the arguments in next test.

    def _create_custom(self):
        from examples.expositional.end2end_apps.custom_app.custom_app import \
            CustomApp

        return CustomApp()

    def _create_basic(self):

        def custom_application(prompt: str) -> str:
            return "a response"

        return custom_application

    def _create_chain(self):
        # Note that while langchain is required, openai is not so tests using
        # this app are optional.

        from langchain.chains import LLMChain
        from langchain.llms.openai import OpenAI
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = OpenAI(temperature=0.0, streaming=False, cache=False)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def _create_feedback_functions(self):
        provider = Dummy(
            loading_prob=0.0,
            freeze_prob=0.0,
            error_prob=0.0,
            overloaded_prob=0.0,
            rpm=1000,
            alloc=1024,  # how much fake data to allocate during requests
        )

        f_dummy1 = Feedback(
            provider.language_match, name="language match"
        ).on_input_output()

        f_dummy2 = Feedback(
            provider.positive_sentiment, name="output sentiment"
        ).on_output()

        f_dummy3 = Feedback(
            provider.positive_sentiment, name="input sentiment"
        ).on_input()

        return [f_dummy1, f_dummy2, f_dummy3]

    def _create_llama(self):
        # Starter example of
        # https://docs.llamaindex.ai/en/latest/getting_started/starter_example.html

        import os

        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import VectorStoreIndex
        os.system(
            'wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -P data/'
        )

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        return query_engine

    def test_required_constructors(self):
        """
        Test the capitilized methods of Tru class that are aliases for various
        app types. This test includes only ones that do not require optional
        packages.
        """
        tru = Tru()

        with self.subTest(type="TruBasicApp"):
            app = self._create_basic()

            with self.subTest(argname=None):
                tru.Basic(app)

            with self.subTest(argname="text_to_text"):
                tru.Basic(text_to_text=app)

            # Not specifying callable should be an error.
            with self.assertRaises(Exception):
                tru.Basic()
            with self.assertRaises(Exception):
                tru.Basic(None)

            # Specifying custom basic app using any of these other argument
            # names should be an error.
            wrong_args = ["app", "chain", "engine"]

            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Basic(**{arg: app})

        with self.subTest(type="TruCustomApp"):
            app = self._create_custom()

            tru.Custom(app)
            tru.Custom(app=app)

            # Not specifying callable should be an error.
            with self.assertRaises(Exception):
                tru.Custom()
            with self.assertRaises(Exception):
                tru.Custom(None)

            # Specifying custom app using any of these other argument names
            # should be an error.
            wrong_args = ["chain", "engine", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Custom(**{arg: app})

        with self.subTest(type="TruVirtual"):
            tru.Virtual(None)

    @optional_test
    def test_optional_constructors(self):
        """
        Test Tru class utility aliases that require optional packages.
        """
        tru = Tru()

        with self.subTest(type="TruChain"):
            app = self._create_chain()

            with self.subTest(argname=None):
                tru.Chain(app)

            with self.subTest(argname="chain"):
                tru.Chain(chain=app)

            # Not specifying chain should be an error.
            with self.assertRaises(Exception):
                tru.Chain()
            with self.assertRaises(Exception):
                tru.Chain(None)

            # Specifying the chain using any of these other argument names
            # should be an error.
            wrong_args = ["app", "engine", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Chain(**{arg: app})

        with self.subTest(type="TruLlama"):
            app = self._create_llama()

            tru.Llama(app)

            tru.Llama(engine=app)

            # Not specifying an engine should be an error.
            with self.assertRaises(Exception):
                tru.Llama()

            with self.assertRaises(Exception):
                tru.Llama(None)

            # Specifying engine using any of these other argument names
            # should be an error.
            wrong_args = ["chain", "app", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Llama(**{arg: app})

    def test_run_feedback_functions_wait(self):
        """
        Test run_feedback_functions in wait mode. This mode blocks until results
        are ready.
        """

        app = self._create_custom()

        feedbacks = self._create_feedback_functions()

        expected_feedback_names = set(f.name for f in feedbacks)

        tru = Tru()

        tru_app = TruCustomApp(app)

        with tru_app as recording:
            response = app.respond_to_query("hello")

        record = recording.get()

        feedback_results = list(
            tru.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=True
            )
        )

        # Check we get the right number of results.
        self.assertEqual(len(feedback_results), len(feedbacks))

        # Check that the results are for the feedbacks we submitted.
        self.assertEqual(
            set(expected_feedback_names),
            set(res.name for res in feedback_results),
            "feedback result names do not match requested feedback names"
        )

        # Check that the structure of returned tuples is correct.
        for result in feedback_results:
            self.assertIsInstance(result, mod_feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        tru.add_feedbacks(feedback_results)

        # Check that results were added to db.
        df, returned_feedback_names = tru.get_records_and_feedback(
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
        expected_feedback_names = set(f.name for f in feedbacks)

        tru = Tru()

        tru_app = TruCustomApp(app)

        with tru_app as recording:
            response = app.respond_to_query("hello")

        record = recording.get()

        start_time = datetime.now()

        future_feedback_results = list(
            tru.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=False
            )
        )

        end_time = datetime.now()

        # Should return quickly.
        self.assertLess(
            (end_time - start_time).total_seconds(),
            2.0,  # TODO: get it to return faster
            "Non-blocking run_feedback_functions did not return fast enough."
        )

        # Check we get the right number of results.
        self.assertEqual(len(future_feedback_results), len(feedbacks))

        feedback_results = []

        # Check that the structure of returned tuples is correct.
        for future_result in future_feedback_results:
            self.assertIsInstance(future_result, FutureClass)

            wait([future_result])

            result = future_result.result()
            self.assertIsInstance(result, mod_feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

            feedback_results.append(result)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        tru.add_feedbacks(feedback_results)

        # Check that results were added to db.
        df, returned_feedback_names = tru.get_records_and_feedback(
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
    #     app_id = "test_app"
    #     app_definition = mod_app_schema.AppDefinition(app_id=app_id, model_dump_json="{}")
    #     tru = Tru()

    #     # Action: Add the app to the database
    #     added_app_id = tru.add_app(app_definition)

    #     # Assert: Verify the app was added successfully
    #     self.assertEqual(app_id, added_app_id)
    #     retrieved_app = tru.get_app(app_id)
    #     self.assertIsNotNone(retrieved_app)
    #     self.assertEqual(retrieved_app['app_id'], app_id)

    # def test_delete_app(self):
    #     # Setup: Add an app to the database
    #     app_id = "test_app"
    #     app_definition = mod_app_schema.AppDefinition(app_id=app_id, model_dump_json="{}")
    #     tru = Tru()
    #     tru.add_app(app_definition)

    #     # Action: Delete the app
    #     tru.delete_app(app_id)

    #     # Assert: Verify the app is deleted
    #     retrieved_app = tru.get_app(app_id)
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

    def test_stop_evaluator(self):
        # TODO
        pass

    def test_stop_dashboard(self):
        # TODO
        pass

    def test_run_dashboard(self):
        pass
