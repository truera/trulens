import logging
from typing import Dict, Optional, Sequence

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint

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

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:

        from litellm import completion as litellm_completion
        if prompt is not None:
            comp = litellm_completion(
                model=self.model_engine,
                messages=[{
                    "role": "system",
                    "content": prompt
                }],
                **kwargs
            )
        elif messages is not None:
            comp = litellm_completion(
                model=self.model_engine, messages=messages, **kwargs
            )

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        assert isinstance(comp, object)

        return comp["choices"][0]["message"]["content"]
