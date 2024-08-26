import json
from typing import ClassVar, Dict, Optional, Sequence

import pydantic
import snowflake
import snowflake.connector
from snowflake.connector import SnowflakeConnection
from snowflake.snowpark import Session
from trulens.feedback import LLMProvider
from trulens.providers.cortex.endpoint import CortexEndpoint


class Cortex(
    LLMProvider
):  # require `pip install snowflake-snowpark-python` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"

    snowflake_session: Session = pydantic.Field(exclude=True)
    model_engine: str
    endpoint: CortexEndpoint = pydantic.Field(exclude=True)
    snowflake_conn: SnowflakeConnection

    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex

    Args:

        connection_parameters (dict): Snowflake connection parameters.
        model_engine (str, optional): Model engine to use. Defaults to `snowflake-arctic`.

        Connecting with user/password:

        !!! example

            ```python
            connection_parameters = {

                "account": <account>,
                "user": <user>,
                "password": <password>,
                "role": <role>,
                "database": <database>,
                "schema": <schema>,
                "warehouse": <warehouse>
            }

            provider = Cortex(connection_parameters)
            ```

        Connecting with private key:

        !!! example

            ```python
            connection_parameters = {

                "account": <account>,
                "user": <user>,
                "private_key": <private_key>,
                "role": <role>,
                "database": <database>,
                "schema": <schema>,
                "warehouse": <warehouse>
            }

        Connecting with a private key file:

        !!! example

            ```python
            connection_parameters = {

                    "account": <account>,
                    "user": <user>,
                    "private_key_file": <private_key_file>,
                    "private_key_file_pwd": <private_key_file_pwd>,
                    "role": <role>,
                    "database": <database>,
                    "schema": <schema>,
                    "warehouse": <warehouse>
                }
            provider = Cortex(connection_parameters)
            ```
    """

    def __init__(
        self,
        connection_parameters: dict,
        model_engine: Optional[str] = None,
        *args,
        **kwargs: Dict,
    ):
        self_kwargs = dict(kwargs)

        self_kwargs["model_engine"] = (
            self.DEFAULT_MODEL_ENGINE if model_engine is None else model_engine
        )

        self_kwargs["endpoint"] = CortexEndpoint(*args, **kwargs)

        # Create a Snowflake connector
        self_kwargs["snowflake_conn"] = snowflake.connector.connect(
            **connection_parameters
        )

        self_kwargs["snowflake_session"] = Session.builder.configs(
            connection_parameters
        ).create()

        super().__init__(**self_kwargs)

    def _exec_snowsql_complete_command(
        self,
        model: str,
        temperature: float,
        messages: Optional[Sequence[Dict]] = None,
    ):
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []

        messages_json_str = json.dumps(messages)

        options = {"temperature": temperature}

        options_json_str = json.dumps(options)

        completion_input_str = """
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                %s,
                parse_json(%s),
                parse_json(%s)
            )
        """
        cursor = self.snowflake_conn.cursor()
        try:
            cursor.execute(
                completion_input_str,
                (model, messages_json_str, options_json_str),
            )
            result = cursor.fetchall()
        finally:
            cursor.close()

        return result

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs,
    ) -> str:
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        if messages is not None:
            kwargs["messages"] = messages

        elif prompt is not None:
            kwargs["messages"] = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        res = self._exec_snowsql_complete_command(**kwargs)

        if len(res) == 0 or len(res[0]) == 0:
            raise ValueError("No completion returned from Snowflake Cortex.")

        completion = json.loads(res[0][0])["choices"][0]["messages"]

        return completion
