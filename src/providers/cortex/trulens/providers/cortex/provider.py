from typing import (
    ClassVar,
    Dict,
    Optional,
    Sequence,
)

from snowflake.cortex import Complete
from snowflake.snowpark import Session
from snowflake.snowpark import context
from snowflake.snowpark.exceptions import SnowparkSessionException
from trulens.core.utils import pyschema as pyschema_utils
from trulens.feedback import llm_provider
from trulens.feedback import prompts as feedback_prompts
from trulens.providers.cortex import endpoint as cortex_endpoint


class Cortex(
    llm_provider.LLMProvider
):  # require `pip install snowflake-snowpark-python snowflake-ml-python>=1.7.1` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_SNOWPARK_SESSION: Optional[Session] = None
    DEFAULT_MODEL_ENGINE: ClassVar[str] = "llama3.1-8b"

    model_engine: str
    endpoint: cortex_endpoint.CortexEndpoint
    snowpark_session: Session

    """Snowflake's Cortex COMPLETE endpoint. Defaults to `llama3.1-8b`.

    Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex

    !!! example

        === "Connecting with user/password"
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
            snowpark_session = Session.builder.configs(connection_parameters).create()
            provider = Cortex(snowpark_session=snowpark_session)
            ```

        === "Connecting with private key"
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
            snowpark_session = Session.builder.configs(connection_parameters).create()
            provider = Cortex(snowpark_session=snowpark_session)
            ```

        === "Connecting with a private key file"
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
            snowpark_session = Session.builder.configs(connection_parameters).create()
            provider = Cortex(snowpark_session=snowpark_session)
            ```

    Args:
        snowpark_session (Session): Snowflake session.

        model_engine (str, optional): Model engine to use. Defaults to `snowflake-arctic`.

    """

    def __init__(
        self,
        snowpark_session: Optional[Session] = None,
        model_engine: Optional[str] = None,
        *args,
        **kwargs: Dict,
    ):
        self_kwargs = dict(kwargs)

        self_kwargs["model_engine"] = (
            self.DEFAULT_MODEL_ENGINE if model_engine is None else model_engine
        )

        self_kwargs["endpoint"] = cortex_endpoint.CortexEndpoint(
            *args, **kwargs
        )

        if snowpark_session is None or pyschema_utils.is_noserio(
            snowpark_session
        ):
            if (
                hasattr(self, "DEFAULT_SNOWPARK_SESSION")
                and self.DEFAULT_SNOWPARK_SESSION is not None
            ):
                snowpark_session = self.DEFAULT_SNOWPARK_SESSION
            else:
                # context.get_active_session() will fail if there is no or more
                # than one active session. This should be fine for server side
                # eval in the warehouse as there should only be one active
                # session in the execution context.
                try:
                    snowpark_session = context.get_active_session()
                except SnowparkSessionException:
                    class_name = (
                        f"{self.__module__}.{self.__class__.__qualname__}"
                    )
                    raise ValueError(
                        "Cannot infer snowpark session to use! Try setting "
                        f"`{class_name}.DEFAULT_SNOWPARK_SESSION`."
                    )
        self_kwargs["snowpark_session"] = snowpark_session

        super().__init__(**self_kwargs)

    def _invoke_cortex_complete(
        self,
        model: str,
        temperature: float,
        messages: Optional[Sequence[Dict]] = None,
    ) -> str:
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []

        options = {"temperature": temperature}

        completion_res_str: str = Complete(
            model=model,
            prompt=messages,
            options=options,
            session=self.snowpark_session,
            stream=False,
        )
        return completion_res_str

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

        completion_str = self._invoke_cortex_complete(**kwargs)

        return completion_str

    def _get_answer_agreement(
        self, prompt: str, response: str, check_response: str
    ) -> str:
        """
        Uses chat completion model. A function that completes a template to
        check if two answers agree.

        Args:
            text (str): A prompt to an agent.
            response (str): The agent's response to the prompt.
            check_response(str): The response to check against.

        Returns:
            str
        """

        assert self.endpoint is not None, "Endpoint is not set."

        messages = [
            {"role": "system", "content": feedback_prompts.AGREEMENT_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": check_response},
        ]

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=messages,
        )
