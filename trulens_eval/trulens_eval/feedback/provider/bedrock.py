import json
import logging
import os
from typing import Dict, Optional, Sequence

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import BedrockEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)


class Bedrock(LLMProvider):
    # LLMProvider requirement which we do not use:
    model_engine: str = "Bedrock"

    model_id: str
    endpoint: BedrockEndpoint

    def __init__(
        self, *args, model_id: str = "amazon.titan-tg1-large", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of AWS Feedback Functions.

        Parameters:

        - model_id (str, optional): The specific model id. Defaults to
          "amazon.titan-tg1-large".

        - All other args/kwargs passed to BedrockEndpoint and subsequently
          to boto3 client constructor.
        """

        # SingletonPerName: return singleton unless client provided
        if hasattr(self, "model_id") and "client" not in kwargs:
            return

        # Pass kwargs to Endpoint. Self has additional ones.
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs['model_id'] = model_id

        self_kwargs['endpoint'] = BedrockEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # LLMProvider requirement
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        assert self.endpoint is not None
        assert prompt is not None, "Bedrock can only operate on `prompt`, not `messages`."

        # NOTE(joshr): only tested with sso auth
        import json

        body = json.dumps({"inputText": prompt})

        modelId = self.model_id

        response = self.endpoint.client.invoke_model(body=body, modelId=modelId)

        response_body = json.loads(response.get('body').read()
                                  ).get('results')[0]["outputText"]
        # text

        return response_body
