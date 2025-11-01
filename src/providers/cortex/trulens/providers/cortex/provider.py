import json
import logging
from typing import ClassVar, Dict, Optional, Sequence, Type, Union

from packaging import version as packaging_version
import pydantic
from snowflake import cortex as cortex_snowflake
from snowflake.ml import version as version_snowflake_ml
from snowflake.snowpark import exceptions as exceptions_snowpark
from snowflake.snowpark import snowpark as snowpark_snowflake
from trulens.core.utils import pyschema as pyschema_utils
from trulens.feedback import llm_provider
from trulens.feedback import prompts as prompts_feedback
from trulens.providers.cortex import endpoint as cortex_endpoint

logger = logging.getLogger(__name__)


class Cortex(
    llm_provider.LLMProvider
):  # require `pip install snowflake-snowpark-python snowflake-ml-python>=1.7.1` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_SNOWPARK_SESSION: Optional[snowpark_snowflake.Session] = None
    DEFAULT_MODEL_ENGINE: ClassVar[str] = "llama3.1-8b"

    model_engine: str
    endpoint: cortex_endpoint.CortexEndpoint
    snowpark_session: snowpark_snowflake.Session
    retry_timeout: Optional[float]

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
            snowpark_session = snowpark_snowflake.Session.builder.configs(connection_parameters).create()
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
            snowpark_session = snowpark_snowflake.Session.builder.configs(connection_parameters).create()
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
            snowpark_session = snowpark_snowflake.Session.builder.configs(connection_parameters).create()
            provider = Cortex(snowpark_session=snowpark_session)
            ```

    Args:
        snowpark_session (snowpark_snowflake.Session): Snowflake session.

        model_engine (str, optional): Model engine to use. Defaults to `snowflake-arctic`.

    """

    def __init__(
        self,
        snowpark_session: Optional[snowpark_snowflake.Session] = None,
        model_engine: Optional[str] = None,
        retry_timeout: Optional[float] = None,
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

        self_kwargs["retry_timeout"] = retry_timeout

        if snowpark_session is None or pyschema_utils.is_noserio(
            snowpark_session
        ):
            if (
                hasattr(self, "DEFAULT_SNOWPARK_SESSION")
                and self.DEFAULT_SNOWPARK_SESSION is not None
            ):
                snowpark_session = self.DEFAULT_SNOWPARK_SESSION
            else:
                # snowpark_snowflake.context.get_active_session() will fail if there is no or more
                # than one active session. This should be fine for server side
                # eval in the warehouse as there should only be one active
                # session in the execution context.
                try:
                    snowpark_session = (
                        snowpark_snowflake.context.get_active_session()
                    )
                except exceptions_snowpark.SnowparkSessionException:
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
        response_format: Optional[Type[pydantic.BaseModel]] = None,
    ) -> Union[str, pydantic.BaseModel]:
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []

        if (
            packaging_version.Version(version_snowflake_ml.VERSION)
            >= packaging_version.Version("1.8.0")
            and response_format is not None
        ):
            options = cortex_snowflake.CompleteOptions(
                temperature=temperature,
                response_format={
                    "type": "json",
                    "schema": response_format.model_json_schema(),
                },
            )
        else:
            options = cortex_snowflake.CompleteOptions(temperature=temperature)

        completion_res: str = cortex_snowflake.complete(
            model=model,
            prompt=messages,
            options=options,
            session=self.snowpark_session,
            stream=False,
            timeout=self.retry_timeout,
        )

        if response_format is not None:
            # If response_format is provided, we expect the response to be a JSON string
            # that can be parsed into the specified response_format.
            try:
                completion_obj = response_format.model_validate_json(
                    completion_res
                )
            except Exception as e:
                logger.debug(
                    f"Parsing response_format {response_format} failed due to {e}, returning raw response.",
                )
                completion_obj = completion_res
        else:
            completion_obj = completion_res

        if packaging_version.Version(
            version_snowflake_ml.VERSION
        ) >= packaging_version.Version("1.7.1"):
            return completion_obj
        # As per https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex#returns,
        # the response is a JSON string with a `choices` key containing an
        # array of completions due to `options` being specified. Currently the
        # array is always of size 1 according to the link.
        return json.loads(completion_res)["choices"][0]["messages"]

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs,
    ) -> Union[str, pydantic.BaseModel]:
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        if response_format is not None:
            kwargs["response_format"] = response_format
        if messages is not None:
            kwargs["messages"] = messages
        elif prompt is not None:
            kwargs["messages"] = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return self._invoke_cortex_complete(**kwargs)

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
            {"role": "system", "content": prompts_feedback.AGREEMENT_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": check_response},
        ]

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=messages,
        )
