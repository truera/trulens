import logging
import os

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import ReplicateEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.keys import set_openai_key
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)


class Replicate(LLMProvider):
    """Out of the box feedback functions calling Replicate API.
    """
    model_engine: str
    endpoint: Endpoint

    def __init__(
        self, *args, endpoint=None, model_engine="andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Create an Replicate Provider with out of the box feedback functions.

        **Usage:**
        ```
        from trulens_eval.feedback.provider.replicate import Replciate
        replicate_provider = Replicate()

        ```

        Args:
            model_engine (str): The Replicate completion model. Defaults to `andreasjansson/llama-2-13b-chat-gguf`
            endpoint (Endpoint): Internal Usage for DB serialization
        """
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self_kwargs['endpoint'] = ReplicateEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__
    
    def _create_chat_completion(self, input, *args, **kwargs):
        import replicate
        return replicate.run(self.model_engine, input={"prompt": input})