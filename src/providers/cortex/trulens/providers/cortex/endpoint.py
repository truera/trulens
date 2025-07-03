import inspect
import json
import logging
import os
import pprint
from typing import Any, Callable, Dict, Optional

from snowflake.cortex._sse_client import Event
from snowflake.cortex._sse_client import SSEClient
from trulens.core.feedback import endpoint as core_endpoint
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class CortexCostComputer:
    @staticmethod
    def handle_response(response: Any) -> Dict[str, Any]:
        model = None
        usage = {}
        for curr in response:
            data = json.loads(curr.data)
            choice = data["choices"][0]
            if "finish_reason" in choice and choice["finish_reason"] == "stop":
                model = data["model"]
                usage = data["usage"]
                break

        if model is None or not usage:
            logger.warning("No model usage found in response.")

        endpoint = CortexEndpoint()
        callback = CortexCallback(endpoint=endpoint)
        return {
            SpanAttributes.COST.MODEL: model,
            SpanAttributes.COST.CURRENCY: "Snowflake credits",
            SpanAttributes.COST.COST: callback._compute_credits_consumed(
                model, usage.get("total_tokens", 0)
            ),
            SpanAttributes.COST.NUM_TOKENS: usage.get("total_tokens", 0),
            SpanAttributes.COST.NUM_PROMPT_TOKENS: usage.get(
                "prompt_tokens", 0
            ),
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: usage.get(
                "completion_tokens", 0
            ),
        }


class CortexCallback(core_endpoint.EndpointCallback):
    _model_costs: Optional[dict] = None
    # TODO (Daniel): cost tracking for Cortex finetuned models is not yet implemented.

    def _compute_credits_consumed(
        self, cortex_model_name: Optional[str], n_tokens: int
    ) -> float:
        try:
            if self._model_costs is None:
                # the credit consumption table needs to be kept up-to-date with
                # the latest cost information https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf#page=9.
                # We should refer to the latest model availability of REST api https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#model-availability

                with open(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "config/cortex_model_costs.json",
                    ),
                    "r",
                ) as file:
                    self._model_costs = json.load(file)

            if cortex_model_name and cortex_model_name in self._model_costs:
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
            if "choices" in response_dict:
                choice = response_dict["choices"][0]
                if (
                    "finish_reason" in choice
                    and choice["finish_reason"] == "stop"
                ):
                    self.global_callback.handle_generation(
                        response=response_dict
                    )

                    if callback is not None:
                        callback.handle_generation(response=response_dict)
        else:
            logger.warning(
                "Unrecognized Cortex response format. It did not have usage information:\n%s",
                pp.pformat(response_dict),
            )

        return response
