import logging
from typing import Dict, Optional, Sequence

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LITELLM

with OptionalImports(messages=REQUIREMENT_LITELLM):
    import litellm
    from litellm import completion

    from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint

# check that the optional imports are not dummies:
OptionalImports(messages=REQUIREMENT_LITELLM).assert_installed(litellm)

logger = logging.getLogger(__name__)


class LiteLLM(LLMProvider):
    """Out of the box feedback functions calling LiteLLM API.


    Create an LiteLLM Provider with out of the box feedback functions.

    Usage:
        ```python
        from trulens_eval.feedback.provider.litellm import LiteLLM
        litellm_provider = LiteLLM()
        ```

    Args:
        model_engine: The LiteLLM completion model.Defaults to
            `gpt-3.5-turbo`
        
        endpoint: Internal Usage for DB serialization.
    """
    model_engine: str
    endpoint: Endpoint

    def __init__(
        self, *args, endpoint: Optional[Endpoint] = None, model_engine: str = "gpt-3.5-turbo", **kwargs
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

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

        if prompt is not None:
            comp = completion(
                model=self.model_engine,
                messages=[{
                    "role": "system",
                    "content": prompt
                }],
                **kwargs
            )
        elif messages is not None:
            comp = completion(
                model=self.model_engine, messages=messages, **kwargs
            )

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        assert isinstance(comp, object)

        return comp["choices"][0]["message"]["content"]
