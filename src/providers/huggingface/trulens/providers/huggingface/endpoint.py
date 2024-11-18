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
from trulens.experimental.otel_tracing import _feature as otel_tracing_feature

logger = logging.getLogger(__name__)


class HuggingfaceCallback(core_endpoint.EndpointCallback):
    def handle_classification(self, response: requests.Response) -> None:
        # Huggingface free inference api doesn't seem to have its own library
        # and the docs say to use `requests`` so that is what we instrument and
        # process to track api calls.

        super().handle_classification(response)

        if response.ok:
            self.cost.n_successful_requests += 1
            content = response.json()

            # Handle case when multiple items returned by hf api
            for item in content:
                self.cost.n_classes += len(item)


if otel_tracing_feature._FeatureSetup.are_optionals_installed():
    from trulens.experimental.otel_tracing.core.feedback import (
        endpoint as experimental_core_endpoint,
    )

    class _WrapperHuggingfaceEndpointCallback(
        experimental_core_endpoint._WrapperEndpointCallback[
            requests.Response, serial_utils.JSON
        ]
    ):
        """EXPERIMENTAL(otel_tracing): process huggingface wrapped calls to
        extract cost information.

        !!! Note
            Huggingface free inference api does not have its own modules and the
            documentation suggests to use `requests`. Therefore, this class
            processes request module responses.
        """

        def on_callable_call(self, bindings, **kwargs):
            super().on_callable_call(bindings, **kwargs)

            url = bindings.arguments["url"]
            if not url.startswith("https://api-inference.huggingface.co"):
                logger.debug(
                    "Unknown huggingface api request: %s. Cost tracking will not be available.",
                    url,
                )
                return

            self.cost.n_classification_requests += 1

        def on_callable_return(
            self, ret: requests.Response, **kwargs
        ) -> requests.Response:
            """Process a returned call."""

            super().on_callable_return(ret=ret, **kwargs)

            bindings = self.bindings

            if "url" not in bindings.arguments:
                return ret

            url = bindings.arguments["url"]
            if not url.startswith("https://api-inference.huggingface.co"):
                return ret

            # TODO: Determine whether the request was a classification or some other
            # type of request. Currently we use huggingface only for classification
            # in feedback but this can change.

            if ret.ok:
                self.on_endpoint_classification(response=ret.json())

            return ret

        def on_endpoint_classification(
            self, response: serial_utils.JSON
        ) -> None:
            """Process a classification response."""

            super().on_endpoint_classification(response)

            if not isinstance(response, Sequence):
                logger.warning("Unexpected response: %s", response)
                return

            # Handle case when multiple items returned by hf api
            for item in response:
                if not isinstance(item, Sequence):
                    logger.warning("Unexpected response item: %s", item)
                else:
                    self.cost.n_classes += len(item)


class HuggingfaceEndpoint(core_endpoint._WithPost, core_endpoint.Endpoint):
    """Huggingface endpoint.

    Instruments the requests.post method for requests to
    "https://api-inference.huggingface.co".
    """

    _experimental_wrapper_callback_class = (
        _WrapperHuggingfaceEndpointCallback
        if otel_tracing_feature._FeatureSetup.are_optionals_installed()
        else None
    )

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
