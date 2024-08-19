import inspect
import json
import logging
import pprint
from typing import Any, Callable, ClassVar, Optional

from snowflake.snowpark import DataFrame
from snowflake.snowpark import Session
from trulens.core.feedback import endpoint as base_endpoint

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class WrapperCortexCallback(base_endpoint.WrappedEndpointCallback):
    """EXPERIMENTAL: otel-tracing"""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def on_endpoint_response(self, response: Any) -> None:
        if isinstance(
            response, DataFrame
        ):  # response is a snowflake dataframe instance
            response: dict = json.loads(response.collect()[0][0])

            if "usage" in response:
                self.on_endpoint_handle_generation(response=response)
            else:
                logger.warning(
                    "Unrecognized Cortex response format. It did not have usage information:\n%s",
                    pp.pformat(response),
                )

    def on_endpoint_generation(self, response: dict) -> None:
        """Get the usage information from Cortex LLM function response's usage field."""
        usage = response["usage"]

        # Increment number of requests.
        super().on_endpoint_generation(response)

        # Assume a response that had usage field was successful. Note at the time of writing 06/12/2024, the usage
        # information from Cortex LLM functions is only available when called via snow SQL. It's not fully supported in
        # Python API such as `from snowflake.cortex import Summarize, Complete, ExtractAnswer, Sentiment, Translate` yet.

        self.cost.n_successful_requests += 1

        for cost_field, cortex_field in [
            ("n_tokens", "total_tokens"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens"),
        ]:
            setattr(
                self.cost,
                cost_field,
                getattr(self.cost, cost_field, 0) + usage.get(cortex_field, 0),
            )


class CortexCallback(base_endpoint.EndpointCallback):
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_generation(self, response: dict) -> None:
        """Get the usage information from Cortex LLM function response's usage field."""
        usage = response["usage"]

        # Increment number of requests.
        super().handle_generation(response)

        # Assume a response that had usage field was successful. Note at the time of writing 06/12/2024, the usage
        # information from Cortex LLM functions is only available when called via snow SQL. It's not fully supported in
        # Python API such as `from snowflake.cortex import Summarize, Complete, ExtractAnswer, Sentiment, Translate` yet.

        self.cost.n_successful_requests += 1

        for cost_field, cortex_field in [
            ("n_tokens", "total_tokens"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens"),
        ]:
            setattr(
                self.cost,
                cost_field,
                getattr(self.cost, cost_field, 0) + usage.get(cortex_field, 0),
            )


class CortexEndpoint(base_endpoint.Endpoint):
    """Snowflake Cortex endpoint."""

    def __init__(self, *args, **kwargs):
        if hasattr(self, "name"):
            # singleton already made
            if len(kwargs) > 0:
                logger.warning(
                    "Ignoring additional kwargs for singleton endpoint %s: %s",
                    self.name,
                    pp.pformat(kwargs),
                )
                self.warning()
            return

        kwargs["name"] = "cortex"
        kwargs["callback_class"] = CortexCallback
        kwargs["wrapper_callback_class"] = WrapperCortexCallback

        super().__init__(*args, **kwargs)
        self._instrument_class(Session, "sql")

    def __new__(cls, *args, **kwargs):
        return super(base_endpoint.Endpoint, cls).__new__(cls, name="cortex")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[base_endpoint.EndpointCallback],
    ) -> None:
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        counted_something = False

        if isinstance(
            response, DataFrame
        ):  # response is a snowflake dataframe instance
            response: dict = json.loads(response.collect()[0][0])

            if "usage" in response:
                counted_something = True

                self.global_callback.handle_generation(response=response)

                if callback is not None:
                    callback.handle_generation(response=response)

            if not counted_something:
                logger.warning(
                    "Unrecognized Cortex response format. It did not have usage information:\n%s",
                    pp.pformat(response),
                )
