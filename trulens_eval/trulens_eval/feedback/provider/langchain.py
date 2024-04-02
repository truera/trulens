import json
import logging
from typing import Dict, Optional, Sequence, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import LangchainEndpoint

logger = logging.getLogger(__name__)


class Langchain(LLMProvider):
    """Out of the box feedback functions using LangChain LLMs and ChatModels

    Create a LangChain Provider with out of the box feedback functions.

    !!! example
    
        ```python
        from trulens_eval.feedback.provider.langchain import Langchain
        from langchain_community.llms import OpenAI

        gpt3_llm = OpenAI(model="gpt-3.5-turbo-instruct")
        langchain_provider = Langchain(chain = gpt3_llm)
        ```

    Args:
        chain: LangChain LLM.
    """

    endpoint: LangchainEndpoint

    def __init__(
        self,
        chain: Union[BaseLLM, BaseChatModel],
        *args,
        model_engine: str = "",
        **kwargs
    ):
        self_kwargs = dict(kwargs)
        self_kwargs["model_engine"] = model_engine or type(chain).__name__
        self_kwargs["endpoint"] = LangchainEndpoint(
            *args, chain=chain, **kwargs
        )

        super().__init__(**self_kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if prompt is not None:
            predict = self.endpoint.chain.predict(prompt, **kwargs)

        elif messages is not None:
            prompt = json.dumps(messages)
            predict = self.endpoint.chain.predict(prompt, **kwargs)

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return predict
