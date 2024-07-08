import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Optional, TypeVar

import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LITELLM

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LITELLM) as opt:
    # Here only so we can throw the proper error if litellm is not installed.
    import litellm

opt.assert_installed(litellm)

T = TypeVar("T")


class LiteLLMCallback(EndpointCallback[T]):

    def on_generation(self, response: Any) -> None:
        super().on_generation(response)

        assert self.cost is not None

        self.cost += mod_base_schema.Cost(
            **{
                cost_field: response.get(litellm_field, 0)
                for cost_field, litellm_field in [
                    ("n_tokens", "total_tokens"),
                    ("n_prompt_tokens", "prompt_tokens"),
                    ("n_completion_tokens", "completion_tokens"),
                ]
            }
        )

    def on_response(self, response: pydantic.BaseModel) -> None:
        """Get the usage information from litellm response's usage field."""

        response = response.model_dump()

        usage = response['usage']

        if usage is not None:
            if self.endpoint.litellm_provider not in [
                "openai", "azure", "bedrock"
            ]:
                # We are already tracking costs from the openai or bedrock endpoint so we
                # should not double count here.

                # Increment number of requests.
                super().on_generation(response=usage)

            elif self.endpoint.litellm_provider not in ["openai"]:
                # The total cost does not seem to be properly tracked except by
                # openai so we can try to use litellm costs for this.

                # TODO: what if it is not a completion?
                from litellm import completion_cost

                super().on_generation(response)
                self.cost.cost += completion_cost(response)

        else:
            logger.warning(
                "Unrecognized litellm response format. It did not have usage information:\n%s",
                pp.pformat(response)
            )

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
