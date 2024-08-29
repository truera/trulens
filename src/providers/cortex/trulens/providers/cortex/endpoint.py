import inspect
import json
import logging
import os
import pprint
from typing import Any, Callable, ClassVar, Optional

from snowflake.connector.cursor import SnowflakeCursor
from snowflake.snowpark import DataFrame
from snowflake.snowpark import Session
from trulens.core.feedback import Endpoint
from trulens.core.feedback import EndpointCallback

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class CortexCallback(EndpointCallback):
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
            ("n_cortext_guardrails_tokens", "guardrails_tokens"),
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
            "snowflake_credits_consumed",
            getattr(self.cost, "snowflake_credits_consumed", 0)
            + self._compute_credits_consumed(
                response["model"], usage.get("total_tokens", 0)
            ),
        )


class CortexEndpoint(Endpoint):
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

        super().__init__(*args, **kwargs)

        # Instrument various methods for usage/cost tracking.
        self._instrument_class(Session, "sql")
        self._instrument_class(SnowflakeCursor, "fetchall")

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="cortex")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback],
    ) -> None:
        counted_something = False

        # response is a snowflake dataframe instance or a list if the response is from cursor.fetchall()
        try:
            if isinstance(response, DataFrame):
                response: dict = json.loads(response.collect()[0][0])
            elif isinstance(response, list):
                response: dict = json.loads(response[0][0])
        except Exception as e:
            logger.error(f"Error occurred while parsing response: {e}")
            raise e

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
