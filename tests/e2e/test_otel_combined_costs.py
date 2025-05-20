import os
from typing import List

import litellm
from openai import OpenAI
from snowflake.cortex import Complete
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _TestApp:
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
        self._openai_client = OpenAI()

    def respond_to_query(
        self, query: str, llm_backends: List[str]
    ) -> List[str]:
        ret = []
        for llm_backend in llm_backends:
            if "cortex" == llm_backend:
                ret.append(
                    Complete(
                        model="mistral-large2",
                        prompt=query,
                        session=self._snowpark_session,
                    )
                )
            if "openai" == llm_backend:
                ret.append(
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
            if "litellm" == llm_backend:
                ret.append(
                    litellm.completion(
                        model="gemini/gemini-2.0-flash-exp",
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
        return ret


class TestOtelCombinedCosts(OtelTestCase):
    def test_uncombinable_costs(self):
        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_app",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What's 21+21?", ["cortex", "openai"]),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 1)
        self._check_costs(
            events.iloc[0]["record_attributes"], "mixed", "mixed", False
        )

    def test_combinable_costs(self):
        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_app",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What's 21+21?", ["cortex", "cortex"]),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 1)
        self._check_costs(
            events.iloc[0]["record_attributes"],
            "mistral-large2",
            "Snowflake credits",
            False,
        )
        self.assertEqual(
            events.iloc[0]["record_attributes"][
                SpanAttributes.COST.NUM_PROMPT_TOKENS
            ]
            % 2,
            0,
        )
