
from typing import ClassVar, Dict, Optional, Sequence

import pydantic
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint.base import Endpoint


class Cortex(LLMProvider):
    # require `pip install snowflake-snowpark-python` and a active Snowflake account with proper privileges
    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"
    
    model_engine: str
    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """



    endpoint: Endpoint

    def __init__(
        self,
        model_engine: Optional[str] = None,

        endpoint: Optional[Endpoint] = None,
        **kwargs: dict    
    ):
        pass
    
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine
        
        
        
        
