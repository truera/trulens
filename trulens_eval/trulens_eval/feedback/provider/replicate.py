import os
import requests
import json
from typing import Dict, Optional, Sequence

import logging

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import ReplicateEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)

class Replicate(LLMProvider):
    model_engine: str
    version: str

    def __init__(
        self, 
        *args, 
        model_engine="mistralai/mistral-7b-instruct-v0.1",
        version="83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70",
        **kwargs
    ):
        """
        Replicate Feedback Function Provider

        Parameters:

        - model_engine: should be of the format: provider/model_name:version
        """
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self_kwargs['version'] = version
        self_kwargs['endpoint'] = ReplicateEndpoint(*args, **kwargs)

        super().__init__(**self_kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:

        assert prompt is not None, "Replicate can only operate on `prompt`, not `messages`."

        import replicate
        output = replicate.run(
            f"{self.model_engine}:{self.version}",
            input={"prompt": prompt,
                   "temperature": 0.1}
        )

        # The predict method returns an iterator, we need to get the full output
        full_output = []
        for item in output:
            # https://replicate.com/mistralai/mistral-7b-instruct-v0.1/versions/83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70/api#output-schema
            full_output.append(item)
        # Join the list of strings into a single string
        output_str = ''.join(full_output)
        # Replace single quotes with double quotes
        output_str = output_str.replace("'", '"')
        return output_str 