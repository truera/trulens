import logging
from typing import Any, TypeVar, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LangchainCallback(EndpointCallback[T]):
    """Process langchain wrapped calls to extract cost information.
    
    !!! WARNING
        There is currently no cost tracking other than the number of requests
        included for langchain calls.
    """


class LangchainEndpoint(Endpoint):
    """LangChain endpoint."""

    # Cannot validate BaseLLM / BaseChatModel as they are pydantic v1 and there
    # is some bug involving their use within pydantic v2.
    # https://github.com/langchain-ai/langchain/issues/10112
    chain: Any  # Union[BaseLLM, BaseChatModel]

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="langchain")

    def __init__(self, chain: Union[BaseLLM, BaseChatModel], *args, **kwargs):
        if chain is None:
            raise ValueError("`chain` must be specified.")

        if not (isinstance(chain, BaseLLM) or isinstance(chain, BaseChatModel)):
            raise ValueError(
                f"`chain` must be of type {BaseLLM.__name__} or {BaseChatModel.__name__}. "
                f"If you are using DEFERRED mode, this may be due to our inability to serialize `chain`."
            )

        kwargs["chain"] = chain
        kwargs["name"] = "langchain"
        kwargs["callback_class"] = LangchainCallback

        super().__init__(*args, **kwargs)
