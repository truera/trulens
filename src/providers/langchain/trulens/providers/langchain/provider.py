import logging
from typing import Dict, Optional, Sequence, Type, Union

from langchain_core import messages as langchain_messages
from langchain_core.language_models import chat_models as langchain_chat_models
from langchain_core.language_models import llms as langchain_llms
import pydantic
from trulens.feedback import llm_provider
from trulens.providers.langchain import endpoint as langchain_endpoint

logger = logging.getLogger(__name__)


def _convert_message(message: Dict) -> langchain_messages.BaseMessage:
    """Convert a message to a LangChain BaseMessage."""
    if "role" not in message or message["role"] == "user":
        return langchain_messages.HumanMessage(content=message["content"])
    return langchain_messages.AIMessage(content=message["content"])


class Langchain(llm_provider.LLMProvider):
    """Out of the box feedback functions using LangChain LLMs and ChatModels

    Create a LangChain Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.langchain import LangChain
        from langchain_community.llms import OpenAI

        gpt3_llm = OpenAI(model="gpt-3.5-turbo-instruct")
        langchain_provider = LangChain(chain = gpt3_llm)
        ```

    Args:
        chain: LangChain LLM.
    """

    endpoint: langchain_endpoint.LangchainEndpoint

    def __init__(
        self,
        chain: Union[
            langchain_llms.BaseLLM, langchain_chat_models.BaseChatModel
        ],
        *args,
        model_engine: str = "",
        **kwargs,
    ):
        self_kwargs = dict(kwargs)
        self_kwargs["model_engine"] = model_engine or type(chain).__name__
        self_kwargs["endpoint"] = langchain_endpoint.LangchainEndpoint(
            *args, chain=chain, **kwargs
        )

        super().__init__(**self_kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs,
    ) -> str:
        if prompt is not None:
            predict = self.endpoint.chain.invoke(prompt, **kwargs)

        elif messages is not None:
            messages = [_convert_message(message) for message in messages]
            predict = self.endpoint.chain.invoke(messages, **kwargs)
            if isinstance(
                self.endpoint.chain, langchain_chat_models.BaseChatModel
            ):
                if not isinstance(predict, langchain_messages.BaseMessage):
                    raise ValueError(
                        "`chain.invoke` did not return a `langchain_messages.BaseMessage` as expected!"
                    )
                predict = predict.content
            elif isinstance(self.endpoint.chain, langchain_llms.BaseLLM):
                if not isinstance(predict, str):
                    raise ValueError(
                        "`chain.invoke` did not return a `str` as expected!"
                    )
            else:
                raise ValueError("Unexpected `chain` type!")

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return predict
