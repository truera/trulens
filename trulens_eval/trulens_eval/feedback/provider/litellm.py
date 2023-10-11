import logging
import os

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.keys import set_openai_key
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)


class LiteLLM(LLMProvider):
    """Out of the box feedback functions calling LiteLLM API.
    """
    model_engine: str
    endpoint: Endpoint

    def __init__(
        self, *args, endpoint=None, model_engine="gpt-3.5-turbo", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Create an LiteLLM Provider with out of the box feedback functions.

        **Usage:**
        ```
        from trulens_eval.feedback.provider.litellm import LiteLLM
        litellm_provider = LiteLLM()

        ```

        Args:
            model_engine (str): The LiteLLM completion model.Defaults to `gpt-3.5-turbo`
            endpoint (Endpoint): Internal Usage for DB serialization
        """
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self_kwargs['endpoint'] = LiteLLMEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__
    
    def _create_chat_completion(self, messages, *args, **kwargs):
        import litellm
        return litellm.completion(messages = messages, *args, **kwargs)