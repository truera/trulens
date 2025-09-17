import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument

from tests.util.otel_test_case import OtelTestCase
from tests.util.snowflake_test_case import SnowflakeTestCase

Json = Dict | List


class TestSnowflakeAIObservability(OtelTestCase, SnowflakeTestCase):
    logger = logging.getLogger(__name__)

    def test_get_external_agent_get_records_and_feedback(self):
        # Create TruSession.
        tru_session = self.get_session(
            "test_get_external_agent_get_records_and_feedback",
            connect_via_snowpark_session=True,
            use_account_event_table=True,
        )

        # Create app.
        class _TestApp:
            @instrument()
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_app",
            app_version="1.0.0",
            connector=tru_session.connector,
            main_method=app.greet,
        )
        # Record and invoke.
        start_time = datetime.datetime.now(datetime.timezone.utc)
        with tru_app:
            app.greet("Kojikun")
        end_time = datetime.datetime.now(datetime.timezone.utc)
        self._validate_num_spans_for_app("test_app", 1)
        # Verify.
        records_df, feedback_cols = tru_session.get_records_and_feedback(
            app_name="test_app"
        )
        self.assertEqual(records_df.shape[0], 1)
        self.assertEqual(feedback_cols, [])
        self.assertEqual(records_df["app_name"].iloc[0], "TEST_APP")
        self.assertEqual(records_df["app_version"].iloc[0], "1.0.0")
        self.assertEqual(records_df["input"].iloc[0], "Kojikun")
        self.assertEqual(records_df["output"].iloc[0], "Hello, Kojikun!")
        # Verify `get_events` directly.
        self.assertEqual(
            1,
            len(
                tru_session.connector.db.get_events(
                    app_name="test_app",
                    start_time=start_time - datetime.timedelta(seconds=10),
                )
            ),
        )
        self.assertEqual(
            0,
            len(
                tru_session.connector.db.get_events(
                    app_name="test_app",
                    start_time=end_time + datetime.timedelta(seconds=10),
                )
            ),
        )

    def test_get_cortex_agent_get_records_and_feedback(self):
        # Create TruSession.
        tru_session = self.get_session(
            "test_get_cortex_agent_get_records_and_feedback",
            connect_via_snowpark_session=True,
            use_account_event_table=True,
        )
        # Create Cortex agent.
        agent = _CortexAgent(
            "test_cortex_agent",
            os.environ["SNOWFLAKE_PAT"],
            self._database,
            self._schema,
        )
        self.assertEqual(
            ["TEST_CORTEX_AGENT"], [curr["name"] for curr in agent.get_agents()]
        )
        # Invoke.
        agent.call("Why is Kojikun so cute?")
        self._validate_num_spans_for_app(
            "test_cortex_agent", 1, app_type="CORTEX AGENT"
        )
        # Verify.
        records_df, feedback_cols = tru_session.get_records_and_feedback(
            app_name="test_cortex_agent"
        )
        self.assertEqual(records_df.shape[0], 1)
        self.assertEqual(feedback_cols, [])
        self.assertEqual(records_df["app_name"].iloc[0], "TEST_CORTEX_AGENT")
        self.assertEqual(
            records_df["app_version"].iloc[0], 2
        )  # TODO(this_pr): not sure what this is about!
        self.assertEqual(records_df["input"].iloc[0], "Kojikun")
        self.assertEqual(records_df["output"].iloc[0], "Hello, Kojikun!")


class _CortexAgent:
    logger = logging.getLogger(__name__)

    def __init__(
        self, name: str, pat_token: str, database: str, schema: str
    ) -> None:
        self._name = name
        self._database = database
        self._schema = schema
        self._pat_token = pat_token
        data = {
            "name": name,
            "comment": "This is a test agent.",
            "models": {"orchestration": "llama3.3-70B"},
        }
        response = self._rest_post("", data)
        if response.status_code != 200:
            raise ValueError(f"Error: {response.status_code}")

    def _rest_call(
        self, url_suffix: str, rest_function: Callable, data: Optional[Json]
    ) -> None:
        pat_token = os.environ["SNOWFLAKE_PAT"]
        account_identifier = os.environ["SNOWFLAKE_ACCOUNT"]
        headers = {
            "Authorization": f"Bearer {pat_token}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Accept": "application/json",
        }
        url = (
            f"https://{account_identifier}.snowflakecomputing.com"
            f"/api/v2/databases/{self._database}/schemas/{self._schema}/agents"
        )
        url += url_suffix
        return rest_function(url, headers=headers, json=data)

    def _rest_get(
        self, url_suffix: str, data: Optional[Any]
    ) -> requests.models.Response:
        return self._rest_call(url_suffix, requests.get, data)

    def _rest_post(
        self, url_suffix: str, data: Optional[Any]
    ) -> requests.models.Response:
        return self._rest_call(url_suffix, requests.post, data)

    def call(self, query: str) -> None:
        url_suffix = f"/{self._name}:run"
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ],
        }
        response = self._rest_post(url_suffix, input_data)
        return self._parse_streamed_response(response)

    def _parse_event(self, line: bytes) -> Json:
        decoded_line = str(line.decode("utf-8"))
        if not decoded_line.startswith("data:"):
            raise ValueError()
        json_str = decoded_line[len("data:") :].strip()
        json_str = json_str.strip()
        return json.loads(json_str)

    def _parse_streamed_response(
        self, response: requests.models.Response
    ) -> Tuple[Json, Json]:
        ret = None
        execution_trace = None
        iterator = response.iter_lines()
        for line in iterator:
            if line:
                decoded_line = str(line.decode("utf-8"))
                if decoded_line == "event: response":
                    if ret is not None:
                        raise ValueError()
                    ret = self._parse_event(next(iterator))
                elif decoded_line == "event: execution_trace":
                    if execution_trace is not None:
                        raise ValueError()
                    execution_trace = self._parse_event(next(iterator))
                else:
                    self.logger.info(f"type: {decoded_line}")
                    next(iterator)
        if ret is None:
            raise ValueError("No response returned!")
        if execution_trace is None:
            raise ValueError("No execution trace returned!")
        return ret, execution_trace

    def get_agents(self) -> Json:
        response = self._rest_get("", None)
        if response.status_code == 200:
            return response.json()
        raise ValueError(f"Error: {response.status_code}!")
