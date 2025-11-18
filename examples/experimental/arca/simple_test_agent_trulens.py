import json
import logging
import os
from pprint import pprint
import time
from typing import Dict, List, Tuple

import requests
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.session import TruSession
from trulens.dashboard import run_dashboard

Json = Dict | List

os.environ["TRULENS_OTEL_TRACING"] = "1"

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_trulens_dashboard() -> None:
    connection_params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PAT"],
        "database": "DKUROKAWA",
        "schema": "RCA_EXPERIMENTS",
        "role": os.environ["SNOWFLAKE_ROLE"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    }
    snowpark_session = Session.builder.configs(connection_params).create()
    connector = SnowflakeConnector(
        snowpark_session=snowpark_session,
        use_account_event_table=True,
    )
    tru_session = TruSession(connector=connector)
    run_dashboard(tru_session)


class Agent:
    pat_token = os.environ["SNOWFLAKE_PAT"]
    account_identifier = os.environ["SNOWFLAKE_ACCOUNT"]
    database_name = "DKUROKAWA"
    schema_name = "RCA_EXPERIMENTS"
    headers = {
        "Authorization": f"Bearer {pat_token}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "Accept": "application/json",
    }

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    @classmethod
    def get_agents(cls) -> Json:
        url = (
            f"https://{cls.account_identifier}.snowflakecomputing.com"
            f"/api/v2/databases/{cls.database_name}/schemas/{cls.schema_name}/agents"
        )
        response = requests.get(url, headers=cls.headers)
        if response.status_code == 200:
            print("Success! Agents found:")
            agents = response.json()
            for agent in agents:
                print(
                    f"  - Name: {agent.get('name')}, Owner: {agent.get('owner')}"
                )
            return agents
        print(f"Error: {response.status_code}")
        print(response.text)
        raise ValueError()

    def call_agent(self, input_data) -> Tuple[Json, Json]:
        url = (
            f"https://{self.account_identifier}.snowflakecomputing.com"
            f"/api/v2/databases/{self.database_name}/schemas/{self.schema_name}/agents/{self.agent_name}:run"
        )
        response = requests.post(url, headers=self.headers, json=input_data)
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
                    logger.info(f"type: {decoded_line}")
                    next(iterator)
        if ret is None or execution_trace is None:
            raise ValueError()
        return ret, execution_trace


agents = Agent.get_agents()
agent = Agent("DKUROKAWA_RCA_EXPERIMENTS")
response, execution_trace = agent.call_agent({
    # "thread_id": "1234567890",
    # "parent_message_id": "1234567890",
    # "tool_choice": {"type": "auto"},
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "which movie with Tina Fey is considered the best",
                }
            ],
        }
    ],
})
# print(agents)
# print(type(response))
# print(response)
# print(type(execution_trace))
# print(execution_trace)
# print(response["content"][-1]["text"])
print("EXECUTION TRACE")
for span in execution_trace:
    if "span_type" in str(span) or "span.type" in str(span):
        print("HI")
    # pprint(span)
    # print()
print("DONE EXECUTION TRACE")

run_trulens_dashboard()

print("DONE")
time.sleep(1000)
