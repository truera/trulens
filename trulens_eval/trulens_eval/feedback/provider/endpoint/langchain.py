import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.pyschema import WithClassInfo

logger = logging.getLogger(__name__)


class LangchainCallback(EndpointCallback):
    class Config:
        arbitrary_types_allowed = True

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class LangchainEndpoint(Endpoint, WithClassInfo):
    """
    Langchain endpoint.
    """

    client: Union[BaseLLM, BaseChatModel]

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

    def __init__(self, *args, **kwargs):
        kwargs["name"] = "langchain"
        kwargs["callback_class"] = LangchainCallback

        client = kwargs.get("client")
        if client is None:
            raise ValueError("`client` must be specified.")

        if not (
            isinstance(client, BaseLLM) or isinstance(client, BaseChatModel)
        ):
            raise ValueError(
                f"`client` must be of type {BaseLLM.__name__} or {BaseChatModel.__name__}"
            )

        kwargs["obj"] = self

        super().__init__(*args, **kwargs)
