import os
from typing import Any, Dict, List
import unittest

from langchain_community.chat_models import ChatSnowflakeCortex
import litellm
from openai import OpenAI
from opentelemetry.util.types import AttributeValue
from snowflake.cortex import Complete
from snowflake.snowpark import Session
from trulens.apps.custom import TruCustomApp
from trulens.apps.langchain import TruChain
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


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

    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    def respond_to_query(self, query: str) -> str:
        return Complete(
            model="mistral-large2",
            prompt=query,
            session=self._snowpark_session,
        )


class _TestOpenAIApp:
    def __init__(self) -> None:
        self._openai_client = OpenAI()

    @instrument(span_type=SpanAttributes.SpanType.MAIN)
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

    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    # @old_instrument
    def respond_to_query(self, query: str) -> str:
        completion = (
            litellm.completion(
                # model="mistral/mistral-small", # TODO(this_pr) test this?
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


class TestOtelCosts(OtelAppTestCase):
    def _check_costs(
        self,
        record_attributes: Dict[str, AttributeValue],
        span_name: str,
        cost_model: str,
        cost_currency: str,
        free: bool,
    ):
        self.assertEqual(
            record_attributes["name"],
            span_name,
        )
        self.assertEqual(
            record_attributes[f"{BASE_SCOPE}.costs.model"],
            cost_model,
        )
        self.assertEqual(
            record_attributes[f"{BASE_SCOPE}.costs.cost_currency"],
            cost_currency,
        )
        if free:
            self.assertEqual(
                record_attributes[f"{BASE_SCOPE}.costs.cost"],
                0,
            )
        else:
            self.assertGreater(
                record_attributes[f"{BASE_SCOPE}.costs.cost"],
                0,
            )
        self.assertGreater(
            record_attributes[f"{BASE_SCOPE}.costs.n_tokens"],
            0,
        )
        self.assertGreater(
            record_attributes[f"{BASE_SCOPE}.costs.n_prompt_tokens"],
            0,
        )
        self.assertGreater(
            record_attributes[f"{BASE_SCOPE}.costs.n_completion_tokens"],
            0,
        )
        self.assertGreater(
            len(record_attributes[f"{BASE_SCOPE}.costs.return"]),
            0,
        )

    def _test_tru_custom_app(
        self,
        app: Any,
        cost_functions: List[str],
        model: str,
        currency: str,
        free: bool = False,
    ):
        # Create app.
        tru_recorder = TruCustomApp(
            app,
            app_name="testing",
            app_version="v1",
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            app.respond_to_query("How is baby Kojikun able to be so cute?")
        # Compare results to expected.
        TruSession().experimental_force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 2 + len(cost_functions))
        for i, cost_function in enumerate(cost_functions):
            record_attributes = events.iloc[-i - 1]["record_attributes"]
            self._check_costs(
                record_attributes,
                cost_function,
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
        with tru_recorder(run_name="test run", input_id="42"):
            app.invoke("How is baby Kojikun able to be so cute?")
        tru_session.experimental_force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)
        # TODO: do some asserts

    def test_tru_custom_app_cortex(self):
        self._test_tru_custom_app(
            _TestCortexApp(),
            ["snowflake.cortex._sse_client.SSEClient.events"],
            "mistral-large2",
            "Snowflake credits",
        )

    def test_tru_custom_app_openai(self):
        self._test_tru_custom_app(
            _TestOpenAIApp(),
            ["openai.resources.chat.completions.Completions.create"],
            "gpt-3.5-turbo-0125",
            "USD",
        )

    def test_tru_custom_app_litellm_openai(self):
        model = "gpt-3.5-turbo-0125"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            [
                "openai.resources.chat.completions.Completions.create",
                "litellm.main.completion",
            ],
            model,
            "USD",
        )

    def test_tru_custom_app_litellm_gemini(self):
        model = "gemini/gemini-2.0-flash-exp"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            ["litellm.main.completion"],
            model,
            "USD",
            free=True,
        )

    # TODO(otel): Get keys for this!
    @unittest.skip("Don't have keys")
    def test_tru_custom_app_litellm_huggingface(self):
        model = "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
        self._test_tru_custom_app(
            _TestLiteLLMApp(model),
            ["litellm.main.completion"],
            model,
            "USD",
        )


if __name__ == "__main__":
    unittest.main()
