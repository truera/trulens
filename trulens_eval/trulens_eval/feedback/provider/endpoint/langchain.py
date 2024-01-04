import inspect
import logging
from typing import Any, Callable, ClassVar, Dict, Optional, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.pyschema import WithClassInfo

logger = logging.getLogger(__name__)


class LangchainCallback(EndpointCallback):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class LangchainEndpoint(Endpoint, WithClassInfo):
    """
    Langchain endpoint.
    """

    chain: Union[BaseLLM, BaseChatModel]

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="langchain")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback],
    ) -> None:
        # TODO: Implement this and wrapped
        self.global_callback.handle_generation(response=None)
        if callback is not None:
            callback.handle_generation(response=None)

    def __init__(self, chain: Union[BaseLLM, BaseChatModel], *args, **kwargs):
        if chain is None:
            raise ValueError("`chain` must be specified.")

        if not (isinstance(chain, BaseLLM) or isinstance(chain, BaseChatModel)):
            raise ValueError(
                f"`chain` must be of type {BaseLLM.__name__} or {BaseChatModel.__name__}"
            )

        kwargs["chain"] = chain
        kwargs["name"] = "langchain"
        kwargs["callback_class"] = LangchainCallback
        kwargs["obj"] = self

        super().__init__(*args, **kwargs)
