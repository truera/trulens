import inspect
import json
import logging
import os
import pprint
from typing import Any, Callable, ClassVar, Optional

from snowflake.cortex._sse_client import Event
from snowflake.cortex._sse_client import SSEClient
from trulens.core.feedback import endpoint as core_endpoint

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class CortexCallback(core_endpoint.EndpointCallback):
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)
    _model_costs: Optional[dict] = None
    # TODO (Daniel): cost tracking for Cortex finetuned models is not yet implemented.

    def _compute_credits_consumed(
        self, cortex_model_name: str, n_tokens: int
    ) -> float:
        try:
            if self._model_costs is None:
                # the credit consumption table needs to be kept up-to-date with
                # the latest cost information https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf#page=9.

                with open(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "config/cortex_model_costs.json",
                    ),
                    "r",
                ) as file:
                    self._model_costs = json.load(file)

            if cortex_model_name in self._model_costs:
                return (
                    self._model_costs[cortex_model_name] * n_tokens / 1e6
                )  # we maintain config per-1M-token cost
            else:
                raise ValueError(
                    f"Model {cortex_model_name} not valid or not supported yet for cost estimation."
                )
        except Exception as e:
            logger.error(
                f"Error occurred while computing credits consumed for model {cortex_model_name}: {e}"
            )
            return 0.0

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
            ("n_cortex_guardrails_tokens", "guardrails_tokens"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens"),
        ]:
            setattr(
                self.cost,
                cost_field,
                getattr(self.cost, cost_field, 0) + usage.get(cortex_field, 0),
            )

        # compute credits consumed in Snowflake account based on tokens processed
        setattr(
            self.cost,
            "cost",
            getattr(self.cost, "cost", 0)
            + self._compute_credits_consumed(
                response["model"], usage.get("total_tokens", 0)
            ),
        )

        setattr(self.cost, "cost_currency", "Snowflake credits")


class CortexEndpoint(core_endpoint.Endpoint):
    """Snowflake Cortex endpoint."""

    def __init__(self, *args, **kwargs):
        kwargs["callback_class"] = CortexCallback

        super().__init__(*args, **kwargs)

        # we instrument the SSEClient class from snowflake.cortex module to get the usage information from the HTTP response when calling the REST Complete endpoint
        self._instrument_class(SSEClient, "events")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        counted_something = False

        response_dict = None, None

        try:
            if isinstance(response, Event):
                response_dict = json.loads(
                    response.data
                )  # response is a server-sent event (SSE). see _sse_client.py from snowflake.cortex module for reference

        except Exception as e:
            logger.error(f"Error occurred while parsing response: {e}")
            raise e

        if isinstance(response_dict, dict) and "usage" in response_dict:
            counted_something = True

            self.global_callback.handle_generation(response=response_dict)

            if callback is not None:
                callback.handle_generation(response=response_dict)

        if not counted_something:
            logger.warning(
                "Unrecognized Cortex response format. It did not have usage information:\n%s",
                pp.pformat(response_dict),
            )

        return response
