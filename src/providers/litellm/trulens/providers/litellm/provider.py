import logging
import os
from typing import ClassVar, Dict, Optional, Sequence, Type

import pydantic
from pydantic import BaseModel
from trulens.core.feedback import endpoint as core_endpoint
from trulens.feedback import llm_provider as llm_provider
from trulens.providers.litellm import endpoint as litellm_endpoint

from litellm import completion
from litellm import get_supported_openai_params

logger = logging.getLogger(__name__)


class LiteLLM(llm_provider.LLMProvider):
    """Out of the box feedback functions calling LiteLLM API.

    Create an LiteLLM Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.litellm import LiteLLM
        litellm_provider = LiteLLM()
        ```
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "gpt-3.5-turbo"

    model_engine: str
    """The LiteLLM completion model. Defaults to `gpt-3.5-turbo`."""

    completion_args: Dict[str, str] = pydantic.Field(default_factory=dict)
    """Additional arguments to pass to the `litellm.completion` as needed for chosen api."""

    endpoint: core_endpoint.Endpoint

    def __init__(
        self,
        model_engine: Optional[str] = None,
        completion_kwargs: Optional[Dict] = None,
        endpoint: Optional[core_endpoint.Endpoint] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        from litellm.utils import get_llm_provider

        litellm_provider = get_llm_provider(model_engine)[1]

        if completion_kwargs is None:
            completion_kwargs = {}

        if model_engine.startswith("azure/") and (
            "api_base" not in completion_kwargs
            and not os.getenv("AZURE_API_BASE")
        ):
            raise ValueError(
                "Azure model engine requires 'api_base' parameter to litellm completions. "
                "Provide it to LiteLLM provider in the 'completion_kwargs' parameter:"
                """
                ```python
                provider = LiteLLM(
                    "azure/your_deployment_name",
                    completion_kwargs={
                        "api_base": "https://yourendpoint.openai.azure.com/"
                    }
                )
                ```
                """
            )

        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine
        self_kwargs["litellm_provider"] = litellm_provider
        self_kwargs["completion_args"] = completion_kwargs
        self_kwargs["endpoint"] = litellm_endpoint.LiteLLMEndpoint(
            litellm_provider=litellm_provider, **kwargs
        )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        completion_args = kwargs
        completion_args["model"] = self.model_engine
        completion_args.update(self.completion_args)

        if messages is not None:
            completion_args["messages"] = messages
        elif prompt is not None:
            completion_args["messages"] = [
                {"role": "system", "content": prompt}
            ]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        if "/" in self.model_engine:
            # if we have model provider and model name, verify params
            _model_provider, _model_name = self.model_engine.split("/", 1)
            required_params = ["model", "messages"]
            supported_params = get_supported_openai_params(
                model=_model_name, custom_llm_provider=_model_provider
            )
            params = required_params + (supported_params or [])
            completion_args = {
                k: v for k, v in completion_args.items() if k in params
            }

        comp = completion(**completion_args)
        return comp.choices[0].message.content
