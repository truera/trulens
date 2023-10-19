import inspect
import logging
import pprint
from typing import Any, Callable, Dict, List, Optional

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.keys import _check_key
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.text import UNICODE_CHECK

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class ReplicateCallback(EndpointCallback):

    class Config:
        arbitrary_types_allowed = True

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class ReplicateEndpoint(Endpoint, WithClassInfo):
    """
    Replicate endpoint. Instruments "completion" methods in replicate.* classes.
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="litellm")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        model_name = ""
        if 'model' in bindings.kwargs:
            model_name = bindings.kwargs['model']

        results = None
        if "results" in response:
            results = response['results']

        counted_something = False

        if 'usage' in response:
            counted_something = True
            usage = response['usage']

            self.global_callback.handle_generation(response=usage)

            if callback is not None:
                callback.handle_generation(response=usage)

        if not counted_something:
            logger.warning(
                f"Unrecognized replicate response format. It did not have usage information:\n"
                + pp.pformat(response)
            )

    def __init__(self, *args, **kwargs):
        import os

        kwargs['name'] = "replicate"
        kwargs['callback_class'] = ReplicateCallback

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)
