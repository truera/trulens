import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Dict, Optional

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LITELLM

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LITELLM):
    # Here only so we can throw the proper error if litellm is not installed.
    import litellm

OptionalImports(messages=REQUIREMENT_LITELLM).assert_installed(litellm)


class LiteLLMCallback(EndpointCallback):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class LiteLLMEndpoint(Endpoint):
    """
    LiteLLM endpoint. Instruments "completion" methods in litellm.* classes.
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="litellm")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        model_name = ""
        if hasattr(response, 'model'):
            model_name = response.model

        counted_something = False
        if hasattr(response, 'usage'):
            counted_something = True
            usage = response.usage.dict()

            self.global_callback.handle_generation(response=usage)

            if callback is not None:
                callback.handle_generation(response=usage)

        if not counted_something:
            logger.warning(
                f"Unrecognized litellm response format. It did not have usage information:\n"
                + pp.pformat(response)
            )

    def __init__(self, *args, **kwargs):
        import os

        kwargs['name'] = "litellm"
        kwargs['callback_class'] = LiteLLMCallback

        super().__init__(*args, **kwargs)
