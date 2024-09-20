import inspect
import json
import logging
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
)

import requests
from trulens.core.feedback import Endpoint
from trulens.core.feedback import EndpointCallback
from trulens.core.utils.keys import _check_key
from trulens.core.utils.keys import get_huggingface_headers
from trulens.core.utils.python import safe_hasattr
from trulens.core.utils.serial import JSON
from trulens.core.utils.threading import DEFAULT_NETWORK_TIMEOUT

logger = logging.getLogger(__name__)


class HuggingfaceCallback(EndpointCallback):
    def handle_classification(self, response: requests.Response) -> None:
        # Huggingface free inference api doesn't seem to have its own library
        # and the docs say to use `requests`` so that is what we instrument and
        # process to track api calls.

        super().handle_classification(response)

        if response.ok:
            self.cost.n_successful_requests += 1
            content = json.loads(response.text)

            # Handle case when multiple items returned by hf api
            for item in content:
                self.cost.n_classes += len(item)


class HuggingfaceEndpoint(Endpoint):
    """Huggingface endpoint.

    Instruments the requests.post method for requests to
    "https://api-inference.huggingface.co".
    """

    def __init__(self, *args, **kwargs):
        if safe_hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        kwargs["name"] = "huggingface"
        kwargs["callback_class"] = HuggingfaceCallback

        # Returns true in "warn" mode to indicate that key is set. Does not
        # print anything even if key not set.
        if _check_key("HUGGINGFACE_API_KEY", silent=True, warn=True):
            kwargs["post_headers"] = get_huggingface_headers()

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="huggingface")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: requests.Response,
        callback: Optional[EndpointCallback],
    ) -> requests.Response:
        # Call here can only be requests.post .

        if "url" not in bindings.arguments:
            return response

        url = bindings.arguments["url"]
        if not url.startswith("https://api-inference.huggingface.co"):
            return response

        # TODO: Determine whether the request was a classification or some other
        # type of request. Currently we use huggingface only for classification
        # in feedback but this can change.

        self.global_callback.handle_classification(response=response)

        if callback is not None:
            callback.handle_classification(response=response)

        return response

    def post(
        self, url: str, payload: JSON, timeout: float = DEFAULT_NETWORK_TIMEOUT
    ) -> Any:
        self.pace_me()
        ret = requests.post(
            url, json=payload, timeout=timeout, headers=self.post_headers
        )

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j["estimated_time"]
            logger.error("Waiting for %s (%s) second(s).", j, wait_time)
            sleep(wait_time + 2)
            return self.post(url, payload)

        elif isinstance(j, Dict) and "error" in j:
            error = j["error"]
            logger.error("API error: %s.", j)

            if error == "overloaded":
                logger.error("Waiting for overloaded API before trying again.")
                sleep(10.0)
                return self.post(url, payload)
            else:
                raise RuntimeError(error)

        assert (
            isinstance(j, Sequence) and len(j) > 0
        ), f"Post did not return a sequence: {j}"

        if len(j) == 1:
            return j[0]

        else:
            return j
