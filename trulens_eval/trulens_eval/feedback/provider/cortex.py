
from typing import ClassVar, Dict, Optional, Sequence

import os
import pydantic
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from snowflake.cortex import Complete

class Cortex(LLMProvider):
    # require `pip install snowflake-snowpark-python` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability
    
    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"
    
    model_engine: str
    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """



    endpoint: Endpoint

    def __init__(
        self,
        model_engine: Optional[str] = None,
        snowflake_connection_params: Optional[Dict] = None,
        endpoint: Optional[Endpoint] = None,
        **kwargs: dict    
    ):
        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE
        
        from snowflake.snowpark import Session
        connection_params = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
        }

        # Create a Snowflake session
        snowflake_session = Session.builder.configs(connection_params).create()


    
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine
        
        
        
        
        
