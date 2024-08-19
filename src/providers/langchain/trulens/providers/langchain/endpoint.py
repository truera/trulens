import inspect
import logging
from typing import Any, Callable, ClassVar, Dict, Optional, TypeVar, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from trulens.core.feedback import endpoint as base_endpoint

T = TypeVar("T")

logger = logging.getLogger(__name__)


class WrapperLangchainCallback(base_endpoint.EndpointCallback[T]):
    """EXPERIMENTAL: otel-tracing

    Process langchain wrapped calls to extract cost information.

    !!! WARNING
        There is currently no cost tracking other than the number of requests
        included for langchain calls.
    """

    # NOTE(piotrm): Everything that we do track here is tracked by the
    # superclass but I'm leaving this noop subclass here for future additions.


class LangchainCallback(base_endpoint.EndpointCallback):
    # TODEP: remove after EXPERIMENTAL: otel-tracing

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class LangchainEndpoint(base_endpoint.Endpoint):
    """
    LangChain endpoint.
    """

    # Cannot validate BaseLLM / BaseChatModel as they are pydantic v1 and there
    # is some bug involving their use within pydantic v2.
    # https://github.com/langchain-ai/langchain/issues/10112
    chain: Any  # Union[BaseLLM, BaseChatModel]

    def __new__(cls, *args, **kwargs):
        kwargs["callback_class"] = LangchainCallback
        kwargs["wrapper_callback_class"] = WrapperLangchainCallback

        return super(base_endpoint.Endpoint, cls).__new__(cls, name="langchain")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[base_endpoint.EndpointCallback],
    ) -> None:
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        # TODO: Implement this and wrapped
        self.global_callback.handle_generation(response=None)
        if callback is not None:
            callback.handle_generation(response=None)

    def __init__(self, chain: Union[BaseLLM, BaseChatModel], *args, **kwargs):
        if chain is None:
            raise ValueError("`chain` must be specified.")

        if not (isinstance(chain, BaseLLM) or isinstance(chain, BaseChatModel)):
            raise TypeError(
                f"`chain` must be of type {BaseLLM.__name__} or {BaseChatModel.__name__}. "
                f"If you are using DEFERRED mode, this may be due to our inability to serialize `chain`."
            )

        kwargs["chain"] = chain
        kwargs["name"] = "langchain"
        kwargs["callback_class"] = LangchainCallback

        super().__init__(*args, **kwargs)
