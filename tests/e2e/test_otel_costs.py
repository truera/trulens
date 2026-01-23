import os
from typing import Any
import unittest

from langchain_community.chat_models import ChatSnowflakeCortex
import litellm
from openai import OpenAI
from snowflake.cortex import Complete
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.apps.langchain import TruChain
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase


class _TestCortexApp:
    def __init__(self):
        self._connection_params = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "role": os.environ.get("SNOWFLAKE_ROLE", "ENGINEER"),
            "database": os.environ.get("SNOWFLAKE_DATABASE"),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        }
        self._snowpark_session = Session.builder.configs(
            self._connection_params
        ).create()

    def respond_to_query(self, query: str) -> str:
        return Complete(
            model="mistral-large2",
            prompt=query,
            session=self._snowpark_session,
            timeout=60,
        )


class _TestOpenAIApp:
    def __init__(self) -> None:
        self._openai_client = OpenAI()

    def respond_to_query(self, query: str) -> str:
        return (
            self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            )
            .choices[0]
            .message.content
        )


class _TestLiteLLMApp:
    def __init__(self, model: str) -> None:
        self._model = model

    def respond_to_query(self, query: str) -> str:
        completion = (
            litellm.completion(
                model=self._model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            )
            .choices[0]
            .message.content
        )
        return completion


class TestOtelCosts(OtelTestCase):
    def _test_tru_custom_app(
        self,
        app: Any,
        model: str,
        currency: str,
        num_expected_spans: int = 1,
        free: bool = False,
    ):
        # Create app.
        tru_recorder = TruApp(
            app,
            app_name="testing",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("How is baby Kojikun able to be so cute?",),
        )
        # Compare results to expected.
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), num_expected_spans)
        self._check_costs(
            events.iloc[-1]["record_attributes"],
            model[(model.find("/") + 1) :],
            currency,
            free,
        )

    # TODO(otel): Fix this test!
    @unittest.skip("Not currently working!")
    def test_tru_chain_cortex(self):
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app
        os.environ["SNOWFLAKE_USERNAME"] = os.environ["SNOWFLAKE_USER"]
        os.environ["SNOWFLAKE_PASSWORD"] = os.environ["SNOWFLAKE_USER_PASSWORD"]
        app = ChatSnowflakeCortex(
            model="mistral-large2",
            cortex_function="complete",
        )
        tru_recorder = TruChain(app, app_name="testing", app_version="v1")
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("How is baby Kojikun able to be so cute?",),
        )
        tru_session.force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)
        # TODO: do some asserts

    def test_tru_custom_app_cortex(self):
        self._test_tru_custom_app(
            _TestCortexApp(),
            "mistral-large2",
            "Snowflake credits",
        )

    def test_tru_custom_app_openai(self):
        self._test_tru_custom_app(
            _TestOpenAIApp(),
            "gpt-3.5-turbo-0125",
            "USD",
        )

    def test_tru_custom_app_litellm_openai(self):
        model = "gpt-3.5-turbo-0125"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            model,
            "USD",
            num_expected_spans=2,
        )

    # TODO(otel): Get keys for this!
    @unittest.skip("Don't have keys")
    def test_tru_custom_app_litellm_gemini(self):
        model = "gemini/gemini-2.0-flash-exp"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            model,
            "USD",
            num_expected_spans=2,
            free=True,
        )

    # TODO(otel): Get keys for this!
    @unittest.skip("Don't have keys")
    def test_tru_custom_app_litellm_huggingface(self):
        model = "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            model,
            "USD",
            num_expected_spans=2,
        )
