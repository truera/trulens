import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Optional

import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LITELLM

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LITELLM):
    # Here only so we can throw the proper error if litellm is not installed.
    import litellm

OptionalImports(messages=REQUIREMENT_LITELLM).assert_installed(litellm)


class LiteLLMCallback(EndpointCallback):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_classification(self, response: pydantic.BaseModel) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: pydantic.BaseModel) -> None:
        """Get the usage information from litellm response's usage field."""

        response = response.model_dump()

        usage = response['usage']

        if self.endpoint.litellm_provider not in ["openai", "azure", "bedrock"]:
            # We are already tracking costs from the openai or bedrock endpoint so we
            # should not double count here.

            # Incremente number of requests.
            super().handle_generation(response)

            # Assume a response that had usage field was successful. Otherwise
            # litellm does not provide success counts unlike openai.
            self.cost.n_successful_requests += 1

            for cost_field, litellm_field in [
                ("n_tokens", "total_tokens"),
                ("n_prompt_tokens", "prompt_tokens"),
                ("n_completion_tokens", "completion_tokens"),
            ]:
                setattr(self.cost, cost_field, usage.get(litellm_field, 0))

        if self.endpoint.litellm_provider not in ["openai"]:
            # The total cost does not seem to be properly tracked except by
            # openai so we can use litellm costs for this.

            from litellm import completion_cost
            setattr(self.cost, "cost", completion_cost(response))


class LiteLLMEndpoint(Endpoint):
    """LiteLLM endpoint."""

    litellm_provider: str = "openai"
    """The litellm provider being used.
    
    This is checked to determine whether cost tracking should come from litellm
    or from another endpoint which we already have cost tracking for. Otherwise
    there will be double counting.
    """

    def __init__(self, litellm_provider: str = "openai", **kwargs):
        if hasattr(self, "name"):
            # singleton already made
            if len(kwargs) > 0:
                logger.warning(
                    "Ignoring additional kwargs for singleton endpoint %s: %s",
                    self.name, pp.pformat(kwargs)
                )
                self.warning()
            return

        kwargs['name'] = "litellm"
        kwargs['callback_class'] = LiteLLMCallback

        super().__init__(litellm_provider=litellm_provider, **kwargs)

        import litellm
        self._instrument_module_members(litellm, "completion")

    def __new__(cls, litellm_provider: str = "openai", **kwargs):
        # Problem here if someone uses litellm with different providers. Only a
        # single one will be made. Cannot make a fix just here as
        # track_all_costs creates endpoints via the singleton mechanism.

        return super(Endpoint, cls).__new__(cls, name="litellm")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        counted_something = False

        if hasattr(response, 'usage'):
            counted_something = True

            self.global_callback.handle_generation(response=response)

            if callback is not None:
                callback.handle_generation(response=response)

        if not counted_something:
            logger.warning(
                "Unrecognized litellm response format. It did not have usage information:\n%s",
                pp.pformat(response)
            )
