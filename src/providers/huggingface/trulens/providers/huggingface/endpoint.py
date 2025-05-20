import inspect
import logging
from time import sleep
from typing import (
    Callable,
    Dict,
    Optional,
    Sequence,
)

import requests
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.utils import keys as key_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import threading as threading_utils

logger = logging.getLogger(__name__)


class HuggingfaceCallback(core_endpoint.EndpointCallback):
    def handle_classification(self, response: requests.Response) -> None:
        # HuggingFace free inference API doesn't seem to have its own library
        # and the docs say to use `requests` so that is what we instrument and
        # process to track API calls.

        super().handle_classification(response)

        if response.ok:
            self.cost.n_successful_requests += 1
            content = response.json()

            # Handle case when multiple items returned by HF API
            for item in content:
                self.cost.n_classes += len(item)


class HuggingfaceEndpoint(core_endpoint._WithPost, core_endpoint.Endpoint):
    """HuggingFace endpoint.

    Instruments the requests.post method for requests to
    "https://api-inference.huggingface.co".
    """

    def __init__(self, *args, **kwargs):
        kwargs["callback_class"] = HuggingfaceCallback

        # Returns true in "warn" mode to indicate that key is set. Does not
        # print anything even if key not set.
        if key_utils._check_key("HUGGINGFACE_API_KEY", silent=True, warn=True):
            kwargs["post_headers"] = key_utils.get_huggingface_headers()

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: requests.Response,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> requests.Response:
        # TODELETE(otel_tracing). Delete once otel_tracing is no longer
        # experimental.

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
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: float = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> requests.Response:
        """Make an http post request to the huggingface api.

        This adds some additional logic beyond WithPost.post to handle
        huggingface-specific responses:

        - Model loading delay.
        - Overloaded API.
        - API error.
        """

        ret = super().post(url, json, timeout)

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j["estimated_time"]
            logger.error("Waiting for %s (%s) second(s).", j, wait_time)
            sleep(wait_time + 2)
            return self.post(url, json)

        elif isinstance(j, Dict) and "error" in j:
            error = j["error"]
            logger.error("API error: %s.", j)

            if error == "overloaded":
                logger.error("Waiting for overloaded API before trying again.")
                sleep(10.0)
                return self.post(url, json)
            else:
                raise RuntimeError(error)

        assert (
            isinstance(j, Sequence) and len(j) > 0
        ), f"Post did not return a sequence: {j}"

        return ret
