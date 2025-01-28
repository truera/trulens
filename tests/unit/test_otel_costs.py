import os
import unittest

from langchain_community.chat_models import ChatSnowflakeCortex
from snowflake.cortex import Complete
from snowflake.snowpark import Session
from trulens.apps.custom import TruCustomApp
from trulens.apps.langchain import TruChain
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.instrument import instrument
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

    # @old_instrument
    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    def respond_to_query(self, query: str) -> str:
        return Complete(
            model="mistral-large2",
            prompt=query,
            session=self._snowpark_session,
        )


class TestOtelCosts(OtelAppTestCase):
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
            # account=os.environ["SNOWFLAKE_ACCOUNT"],
            # username=os.environ["SNOWFLAKE_USER"],
            # password=os.environ["SNOWFLAKE_USER_PASSWORD"],
            # database=os.environ["SNOWFLAKE_DATABASE"],
            # schema=os.environ["SNOWFLAKE_SCHEMA"],
            # role=os.environ["SNOWFLAKE_ROLE"],
            # warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        )
        # # Sample input prompt
        # messages = [
        #     SystemMessage(content="You are a friendly assistant."),
        #     HumanMessage(content="What are large language models?"),
        # ]
        # # Invoke the stream method and print each chunk as it arrives
        # print("Stream Method Response:")
        #
        # for chunk in app._stream(messages):
        #     print(chunk.message.content)
        tru_recorder = TruChain(app, app_name="testing", app_version="v1")
        with tru_recorder(run_name="test run", input_id="42"):
            app.invoke("How is baby Kojikun able to be so cute?")
        tru_session.experimental_force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)
        # TODO: do some asserts

    def test_custom_cortex(self):
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app.
        app = _TestCortexApp()
        tru_recorder = TruCustomApp(
            app,
            app_name="testing",
            app_version="v1",
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            app.respond_to_query("How is baby Kojikun able to be so cute?")
        # Compare results to expected.
        tru_session.experimental_force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)
        record_attributes = events.iloc[-1]["record_attributes"]
        self.assertEqual(
            record_attributes["name"],
            "snowflake.cortex._sse_client.SSEClient.events",
        )
        self.assertEqual(
            record_attributes["ai_observability.costs.model"],
            "mistral-large2",
        )
        self.assertEqual(
            record_attributes["ai_observability.costs.cost_currency"],
            "Snowflake credits",
        )
        self.assertGreater(
            record_attributes["ai_observability.costs.cost"],
            0,
        )
        self.assertGreater(
            record_attributes["ai_observability.costs.n_tokens"],
            0,
        )
        self.assertGreater(
            record_attributes["ai_observability.costs.n_tokens"],
            0,
        )
        self.assertGreater(
            record_attributes["ai_observability.costs.n_completion_tokens"],
            0,
        )
        self.assertGreater(
            len(record_attributes["ai_observability.costs.return"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
