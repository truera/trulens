"""
Tests of various functionalities of the `Tru` class.
"""

import os
from pathlib import Path
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest import main
from unittest import TestCase

from examples.expositional.end2end_apps.custom_app.custom_app import CustomApp
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from tests.unit.test import JSONTestCase
from tests.unit.test import optional_test

from trulens_eval import Feedback
from trulens_eval import Tru
from trulens_eval import TruCustomApp
from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.keys import check_keys
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import Record
from trulens_eval.tru_basic_app import TruBasicApp
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.json import jsonify


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
                                self.assertTrue(Path("default_file.db").exists())
                            else:
                                self.assertTrue(Path("default.sqlite").exists())

                        # Need to delete singleton after test as otherwise we
                        # cannot change the arguments in next test.

    def test_required_constructors(self):
        """
        Test the capitilized methods of Tru class that are aliases for various
        app types. This test includes only ones that do not require optional
        packages.
        """
        tru = Tru()

        with self.subTest(type="TruChain"):
            
            prompt = PromptTemplate.from_template(
                """Honestly answer this question: {question}."""
            )
            llm = OpenAI(temperature=0.0, streaming=False, cache=False)

            chain = LLMChain(llm=llm, prompt=prompt)

            with self.subTest(argname=None):
                tru.Chain(chain)

            with self.subTest(argname="chain"):
                tru.Chain(chain=chain)

            # Not specifying chain should be an error.
            with self.assertRaises(Exception):
                tru.Chain()

            # Specifying custom chain using any of these other argument names
            # should be an error.
            wrong_args = ["app", "engine", "text_to_text"]
            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Chain(**{arg: chain})

        with self.subTest(type="TruBasicApp"):
            def custom_application(prompt: str) -> str:
                return "a response"
            
            with self.subTest(argname=None):
                tru.Basic(custom_application)

            with self.subTest(argname="text_to_text"):
                tru.Basic(text_to_text=custom_application)

            # Not specifying callable should be an error.
            with self.assertRaises(Exception):
                tru.Basic()

            # Specifying custom callable using any of these other argument names
            # should be an error.
            wrong_args = ["app", "chain", "engine"]

            for arg in wrong_args:
                with self.subTest(argname=arg):
                    with self.assertRaises(Exception):
                        tru.Basic(**{arg: custom_application})
    
        with self.subTest(type="TruCustomApp"):
            ca = CustomApp()

            ta1 = tru.Custom(ca)
            ta2 = tru.Custom(app=ca)


        with self.subTest(type="TruVirtual"):
            tru.Virtual(None)

    @optional_test
    def test_optional_constructors(self):
        """
        Test Tru class utility aliases that require optional packages.
        """
        tru = Tru()

        with self.subTest(type="TruLlama"):
            # Not specifying an app should be an error.
            with self.assertRaises(Exception):
                tru.Llama(None)

    def test_reset_database(self):
        pass

    def test_add_record(self):
        pass

    def test_add_app(self):
        pass

    def test_add_feedback(self):
        pass

    def test_add_feedbacks(self):
        pass

    def test_get_records_and_feedback(self):
        pass

    def test_get_leaderboard(self):
        pass

    def test_start_evaluator(self):
        pass

    def test_stop_evaluator(self):
        pass

    def test_stop_dashboard(self):
        pass

    def test_run_dashboard(self):
        pass

    def test_run_feedback_functions(self):
        pass
