import inspect
import json
from typing import Callable, Optional

import requests

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.keys import _check_key
from trulens_eval.keys import get_huggingface_headers
from trulens_eval.utils.pyschema import WithClassInfo


class HuggingfaceCallback(EndpointCallback):

    def handle_classification(self, response: requests.Response) -> None:
        # Huggingface free inference api doesn't seem to have its own library
        # and the docs say to use `requests`` so that is what we instrument and
        # process to track api calls.

        super().handle_classification(response)

        if response.ok:
            self.cost.n_successful_requests += 1
            content = json.loads(response.text)

            # Huggingface free inference api for classification returns a list
            # with one element which itself contains scores for each class.
            self.cost.n_classes += len(content[0])


class HuggingfaceEndpoint(Endpoint, WithClassInfo):
    """
    Huggingface. Instruments the requests.post method for requests to
    "https://api-inference.huggingface.co".
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="huggingface")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments,
        response: requests.Response, callback: Optional[EndpointCallback]
    ) -> None:
        # Call here can only be requests.post .

        if "url" not in bindings.arguments:
            return

        url = bindings.arguments['url']
        if not url.startswith("https://api-inference.huggingface.co"):
            return

        # TODO: Determine whether the request was a classification or some other
        # type of request. Currently we use huggingface only for classification
        # in feedback but this can change.

        self.global_callback.handle_classification(response=response)

        if callback is not None:
            callback.handle_classification(response=response)

    def __init__(self, *args, **kwargs):
        if hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        kwargs['name'] = "huggingface"
        kwargs['callback_class'] = HuggingfaceCallback

        # Returns true in "warn" mode to indicate that key is set. Does not
        # print anything even if key not set.
        if _check_key("HUGGINGFACE_API_KEY", silent=True, warn=True):
            kwargs['post_headers'] = get_huggingface_headers()

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")
