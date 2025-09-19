import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union

from langchain_core.language_models import chat_models as langchain_chat_models
from langchain_core.language_models import llms as langchain_llms
from trulens.core.feedback import endpoint as core_endpoint

logger = logging.getLogger(__name__)


class LangchainCallback(core_endpoint.EndpointCallback):
    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class LangchainEndpoint(core_endpoint.Endpoint):
    """
    LangChain endpoint.
    """

    # Cannot validate BaseLLM / BaseChatModel as they are pydantic v1 and there
    # is some bug involving their use within pydantic v2.
    # https://github.com/langchain-ai/langchain/issues/10112
    chain: Any  # Union[langchain_llms.BaseLLM, langchain_chat_models.BaseChatModel]

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> None:
        # TODO: Implement this and wrapped
        self.global_callback.handle_generation(response=None)
        if callback is not None:
            callback.handle_generation(response=None)

    def __init__(
        self,
        chain: Union[
            langchain_llms.BaseLLM, langchain_chat_models.BaseChatModel
        ],
        *args,
        **kwargs,
    ):
        if chain is None:
            raise ValueError("`chain` must be specified.")

        if not (
            isinstance(chain, langchain_llms.BaseLLM)
            or isinstance(chain, langchain_chat_models.BaseChatModel)
        ):
            raise ValueError(
                f"`chain` must be of type {langchain_llms.BaseLLM.__name__} or {langchain_chat_models.BaseChatModel.__name__}. "
                f"If you are using DEFERRED mode, this may be due to our inability to serialize `chain`."
            )

        kwargs["chain"] = chain
        kwargs["callback_class"] = LangchainCallback

        super().__init__(*args, **kwargs)
