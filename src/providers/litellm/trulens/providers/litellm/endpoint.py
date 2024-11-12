import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Optional

import pydantic
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.experimental.otel_tracing import _feature as otel_tracing_feature

import litellm
from litellm import completion_cost

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class LiteLLMCallback(core_endpoint.EndpointCallback):
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_classification(self, response: pydantic.BaseModel) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: pydantic.BaseModel) -> None:
        """Get the usage information from litellm response's usage field."""

        response = response.model_dump()

        usage = response["usage"]

        self.endpoint: LiteLLMEndpoint
        if self.endpoint.litellm_provider not in ["openai", "azure", "bedrock"]:
            # We are already tracking costs from the openai or bedrock endpoint so we
            # should not double count here.

            # Increment number of requests.
            super().handle_generation(response)

            # Assume a response that had usage field was successful. Otherwise
            # litellm does not provide success counts unlike openai.
            self.cost.n_successful_requests += 1

            for cost_field, litellm_field in [
                ("n_tokens", "total_tokens"),
                ("n_prompt_tokens", "prompt_tokens"),
                ("n_completion_tokens", "completion_tokens"),
            ]:
                setattr(
                    self.cost,
                    cost_field,
                    getattr(self.cost, cost_field, 0)
                    + usage.get(litellm_field, 0),
                )

        if self.endpoint.litellm_provider not in ["openai"]:
            # The total cost does not seem to be properly tracked except by
            # openai so we can use litellm costs for this.

            setattr(self.cost, "cost", completion_cost(response))


if otel_tracing_feature._FeatureSetup.are_optionals_installed():
    from trulens.experimental.otel_tracing.core.feedback import (
        endpoint as experimental_core_endpoint,
    )

    class _WrapperLiteLLMEndpointCallback(
        experimental_core_endpoint._WrapperEndpointCallback
    ):
        """EXPERIMENTAL(otel_tracing): process litellm wrapped calls to extract
        cost information."""

    def on_endpoint_response(self, response: Any) -> None:
        """Process a returned call."""

        if not isinstance(response, pydantic.BaseModel):
            raise ValueError(
                f"Expected pydantic model but got {type(response)}: {response}"
            )

        response = response.model_dump()

        usage = response["usage"]

        self.endpoint: LiteLLMEndpoint

        if usage is not None:
            if self.endpoint.litellm_provider not in [
                "openai",
                "azure",
                "bedrock",
            ]:
                # We are already tracking costs from the openai or bedrock endpoint so we
                # should not double count here.

                # Increment number of requests.
                super().on_endpoint_generation(response=usage)

            elif self.endpoint.litellm_provider not in ["openai"]:
                # The total cost does not seem to be properly tracked except by
                # openai so we can try to use litellm costs for this.

                # TODO: what if it is not a completion?
                super().on_endpoint_generation(response)
                self.cost.cost += completion_cost(response)

        else:
            logger.warning(
                "Unrecognized litellm response format. It did not have usage information:\n%s",
                pp.pformat(response),
            )

    def on_endpoint_generation(self, response: Any) -> None:
        """Process a generation/completion."""

        super().on_endpoint_generation(response)

        assert self.cost is not None

        self.cost += base_schema.Cost(**{
            cost_field: response.get(litellm_field, 0)
            for cost_field, litellm_field in [
                ("n_tokens", "total_tokens"),
                ("n_prompt_tokens", "prompt_tokens"),
                ("n_completion_tokens", "completion_tokens"),
            ]
        })


class LiteLLMEndpoint(core_endpoint.Endpoint):
    """LiteLLM endpoint."""

    litellm_provider: str = "openai"
    """The litellm provider being used.

    This is checked to determine whether cost tracking should come from litellm
    or from another endpoint which we already have cost tracking for. Otherwise
    there will be double counting.
    """

    _experimental_wrapper_callback_class = (
        _WrapperLiteLLMEndpointCallback
        if otel_tracing_feature._FeatureSetup.are_optionals_installed()
        else None
    )

    def __init__(self, litellm_provider: str = "openai", **kwargs):
        kwargs["callback_class"] = LiteLLMCallback

        super().__init__(litellm_provider=litellm_provider, **kwargs)

        self._instrument_module_members(litellm, "completion")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> None:
        # TODELETE(otel_tracing). Delete once otel_tracing is no longer
        # experimental.

        counted_something = False

        if hasattr(response, "usage"):
            counted_something = True

            self.global_callback.handle_generation(response=response)

            if callback is not None:
                callback.handle_generation(response=response)

        if not counted_something:
            logger.warning(
                "Unrecognized litellm response format. It did not have usage information:\n%s",
                pp.pformat(response),
            )
