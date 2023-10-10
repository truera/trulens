import logging
import os
from typing import Dict, Optional, Sequence

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import BedrockEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.generated import re_0_10_rating

import json

logger = logging.getLogger(__name__)


class Bedrock(LLMProvider):
    model_id: str
    region_name: str

    def __init__(
        self, *args, model_id="amazon.titan-tg1-large", region_name="us-east-1", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of AWS Feedback Functions.

        Parameters:

        - model_id (str, optional): The specific model id. Defaults to
          "amazon.titan-tg1-large".
        - region_name (str, optional): The specific AWS region name. Defaults to
          "us-east-1"

        - All other args/kwargs passed to the boto3 client constructor.
        """
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs['model_id'] = model_id
        self_kwargs['region_name'] = region_name
        self_kwargs['endpoint'] = BedrockEndpoint(region_name = region_name, *args, **kwargs)

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

        # NOTE(joshr): only tested with sso auth
        import boto3
        import json
        bedrock = boto3.client(service_name='bedrock-runtime')

        assert prompt is not None, "Bedrock can only operate on `prompt`, not `messages`."

        body = json.dumps({"inputText": prompt})

        modelId = self.model_id

        response = bedrock.invoke_model(body=body, modelId=modelId)

        response_body = json.loads(response.get('body').read()).get('results')[0]["outputText"]
        # text
        return response_body