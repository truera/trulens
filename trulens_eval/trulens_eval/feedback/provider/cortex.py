from typing import ClassVar, Dict, Optional, Sequence

import os
import json
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint.cortex import CortexEndpoint
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_CORTEX

with OptionalImports(messages=REQUIREMENT_CORTEX):
    from snowflake.snowpark import Session


class Cortex(LLMProvider):
    # require `pip install snowflake-snowpark-python` and an active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"

    model_engine: str
    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """

    endpoint: CortexEndpoint
    snowflake_session: Session

    def __init__(
        self,
        model_engine: Optional[str] = None,
        snowflake_connection_params: Optional[Dict] = None,
        *args,
        **kwargs: dict
    ):
        self_kwargs = dict(kwargs)

        self_kwargs['model_engine'] = self.DEFAULT_MODEL_ENGINE if model_engine is None else model_engine
        self_kwargs["endpoint"] = CortexEndpoint(
            *args, **kwargs
        )

        connection_params = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
        } if snowflake_connection_params is None else snowflake_connection_params

        # Create a Snowflake session
        self_kwargs['snowflake_session'] = Session.builder.configs(
            connection_params).create()

        super().__init__(**self_kwargs)

    def _escape_string_for_sql(self, input_string: str) -> str:
        escaped_string = input_string.replace('\\', '\\\\')
        escaped_string = escaped_string.replace("'", "''")
        return escaped_string

    def _exec_snowsql_complete_command(self, model: str, temperature: float, messages: Optional[Sequence[Dict]] = None):
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []
        messages_json_str = json.dumps(messages)

        options = {'temperature': temperature}
        options_json_str = json.dumps(options)

        completion_input_str = f"""SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            parse_json('{self._escape_string_for_sql(messages_json_str)}'),
            parse_json('{self._escape_string_for_sql(options_json_str)}')
        )"""

        # Executing Snow SQL command requires an active snow session
        return self.snowflake_session.sql(completion_input_str).collect()

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.0

        if messages is not None:
            kwargs['messages'] = messages

        elif prompt is not None:
            kwargs['messages'] = [
                {
                    "role": "system",
                    "content": prompt
                }
            ]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        res = self._exec_snowsql_complete_command(**kwargs)

        completion = json.loads(res[0][0])["choices"][0]["messages"]

        return completion
