
from typing import ClassVar, Dict, Optional, Sequence

import os
import pydantic
from trulens_eval.feedback.provider.base import LLMProvider
from snowflake.snowpark import Session
from trulens_eval.feedback.provider.endpoint.cortex import CortexEndpoint


class Cortex(LLMProvider):
    # require `pip install snowflake-snowpark-python` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability
    
    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"
    
    model_engine: str
    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """

    endpoint: CortexEndpoint

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
        self_kwargs['snowflake_session'] = Session.builder.configs(connection_params).create()
        
        super().__init__(**self_kwargs)
        
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine
    
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:

        completion_args = kwargs
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine

        completion_args.update(self.completion_args)

        if messages is not None:
            completion_args['messages'] = messages

        elif prompt is not None:
            completion_args['messages'] = [
                {
                    "role": "system",
                    "content": prompt
                }
            ]

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        comp = completion(**completion_args)

        assert isinstance(comp, object)

        return comp["choices"][0]["message"]["content"]
        
        
        
        
