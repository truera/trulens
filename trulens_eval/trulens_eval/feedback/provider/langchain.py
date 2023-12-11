import json
import logging
from typing import Dict, Optional, Sequence, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import LangchainEndpoint

logger = logging.getLogger(__name__)


class Langchain(LLMProvider):
    """Out of the box feedback functions using Langchain LLMs"""

    endpoint: LangchainEndpoint

    def __init__(
        self,
        client: Union[BaseLLM, BaseChatModel],
        model_engine: str = "",
        *args,
        **kwargs
    ):
        """
        Create a Langchain Provider with out of the box feedback functions.

        **Usage:**
        ```
        from trulens_eval.feedback.provider.langchain import Langchain
        from langchain.llms import OpenAI

        gpt3_llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-instruct")
        langchain_provider = Langchain(client = client)
        ```

        Args:
            client (Union[BaseLLM, BaseChatModel]): Langchain LLMs or chat models
        """
        self_kwargs = kwargs.copy()
        self_kwargs["client"] = client
        self_kwargs["model_engine"] = model_engine or type(client).__name__
        self_kwargs["endpoint"] = LangchainEndpoint(*args, **self_kwargs)

        super().__init__(**self_kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if prompt is not None:
            predict = self.endpoint.client.predict(prompt, **kwargs)

        elif messages is not None:
            prompt = json.dumps(messages)
            predict = self.endpoint.client.predict(prompt, **kwargs)

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return predict
