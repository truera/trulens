import json
from typing import TypeVar

import requests

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.keys import _check_key
from trulens_eval.keys import get_huggingface_headers
from trulens_eval.utils.python import safe_hasattr

T = TypeVar("T")


class HuggingfaceCallback(EndpointCallback[T]):
    """Process huggingface wrapped calls to extract cost information.
    
    !!! Note
        Huggingface free inference api does not have its own modules and the
        documentation suggests to use `requests`. Therefore, this class
        processes request module responses.
    """

    def on_endpoint_response(self, response: requests.Response) -> None:
        """Process a returned call."""

        super().on_endpoint_response(response)

        bindings = self.bindings

        if "url" not in bindings.arguments:
            return

        url = bindings.arguments['url']
        if not url.startswith("https://api-inference.huggingface.co"):
            return

        # TODO: Determine whether the request was a classification or some other
        # type of request. Currently we use huggingface only for classification
        # in feedback but this can change.

        if response.ok:
            content = json.loads(response.text)
            self.on_endpoint_classification(response=content)

    def on_endpoint_classification(self, response: dict) -> None:
        """Process a classification response."""

        super().on_endpoint_classification(response)

        # Handle case when multiple items returned by hf api
        for item in response:
            self.cost.n_classes += len(item)


class HuggingfaceEndpoint(Endpoint):
    """Huggingface endpoint.
    
    !!! Instruments
        - module `requests` method `post` when making requests to
        urls starting with "https://api-inference.huggingface.co".
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="huggingface")

    def __init__(self, *args, **kwargs):
        if safe_hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        kwargs['name'] = "huggingface"
        kwargs['callback_class'] = HuggingfaceCallback

        # Returns true in "warn" mode to indicate that key is set. Does not
        # print anything even if key not set.
        if _check_key("HUGGINGFACE_API_KEY", silent=True, warn=True):
            kwargs['post_headers'] = get_huggingface_headers()

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")
