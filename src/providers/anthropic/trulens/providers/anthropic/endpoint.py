from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
import os
import pprint
from typing import (
    Any,
    ClassVar,
    Optional,
)

from litellm import model_cost
import pydantic
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.otel.semconv.trace import SpanAttributes

import anthropic

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

LITELLM_MODEL_COSTS_TABLE = model_cost


def _get_env_api_key() -> str | None:
    """Gets the API key from ANTHROPIC_API_KEY environment variable."""
    return os.environ.get("ANTHROPIC_API_KEY", None)


def _get_model_pricing(model_name: str) -> Optional[tuple[float, float]]:
    """Get per-token input and output pricing from LiteLLM.

    Returns None when the model is not in the pricing table so that callers
    can leave any already-recorded cost alone rather than stamping a
    misleading zero.  This mirrors the behaviour of the OpenAI endpoint's
    ``token_cost_for_model`` helper.

    Reference: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    """
    if not model_name:
        return None

    pricing = LITELLM_MODEL_COSTS_TABLE.get(model_name)
    if pricing is None:
        # Prefix fallback: find all Anthropic entries whose key starts with
        # model_name, then pick the shortest (closest) match to minimise the
        # risk of accidentally using a different model's price.
        matches = [
            (key, value)
            for key, value in LITELLM_MODEL_COSTS_TABLE.items()
            if key.startswith(model_name)
            and value.get("litellm_provider") == "anthropic"
        ]
        if matches:
            matched_key, pricing = min(matches, key=lambda kv: len(kv[0]))
            if len(matches) > 1:
                logger.debug(
                    "Model %s matched %d pricing entries; using %s",
                    model_name,
                    len(matches),
                    matched_key,
                )
        else:
            pricing = None

    if pricing is None:
        logger.warning("Model %s not found in LiteLLM pricing data", model_name)
        return None

    return (
        pricing.get("input_cost_per_token", 0.0),
        pricing.get("output_cost_per_token", 0.0),
    )


class AnthropicCostComputer:
    """Computes cost and token usage from Anthropic API responses."""

    @staticmethod
    def handle_response(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        model_name = getattr(response, "model", "") or ""

        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        result: dict[str, Any] = {
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.NUM_TOKENS: total_tokens,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: input_tokens,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: output_tokens,
            SpanAttributes.COST.NUM_REASONING_TOKENS: 0,
            SpanAttributes.COST.MODEL: model_name,
        }

        pricing = _get_model_pricing(model_name)
        if pricing is not None:
            input_price, output_price = pricing
            cost = input_tokens * input_price + output_tokens * output_price
            result[SpanAttributes.COST.COST] = round(cost, 8)

        return result


def _build_cost(cost_info: dict[str, Any]) -> base_schema.Cost:
    """Build a Cost from the dict returned by AnthropicCostComputer.handle_response.

    SpanAttributes.COST.COST is absent when the model was not found in the
    pricing table; in that case the returned Cost has cost=0.0 (the field
    default) so the caller's running total is not polluted by a fake zero.
    """
    addl_cost = base_schema.Cost(
        cost_currency=cost_info.get(SpanAttributes.COST.CURRENCY, "USD"),
        n_tokens=cost_info.get(SpanAttributes.COST.NUM_TOKENS, 0),
        n_prompt_tokens=cost_info.get(SpanAttributes.COST.NUM_PROMPT_TOKENS, 0),
        n_completion_tokens=cost_info.get(
            SpanAttributes.COST.NUM_COMPLETION_TOKENS, 0
        ),
        n_reasoning_tokens=cost_info.get(
            SpanAttributes.COST.NUM_REASONING_TOKENS, 0
        ),
    )
    cost_val = cost_info.get(SpanAttributes.COST.COST)
    if cost_val is not None:
        addl_cost = addl_cost.model_copy(update={"cost": cost_val})
    return addl_cost


class AnthropicCallback(core_endpoint.EndpointCallback):
    """Callback for Anthropic endpoint instrumentation."""

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)
        cost_info = AnthropicCostComputer.handle_response(response)
        self.cost += _build_cost(cost_info)


class AnthropicClient(pydantic.BaseModel):
    """A serializable wrapper for the Anthropic client.

    This mirrors OpenAIClient in the OpenAI endpoint module — the actual
    client is stored in the ``client`` field (excluded from serialization)
    and other attributes delegate to the wrapped client.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    client: anthropic.Anthropic = pydantic.Field(exclude=True)
    """The wrapped Anthropic client instance."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        if client is None:
            api_key = api_key or _get_env_api_key()
            client = anthropic.Anthropic(api_key=api_key, **kwargs)
        super().__init__(client=client)


class AnthropicEndpoint(core_endpoint.Endpoint):
    """Anthropic endpoint for TruLens instrumentation.

    Wraps the Anthropic Python SDK client and handles pacing, cost tracking,
    and OpenTelemetry integration.

    Args:
        client: An anthropic.Anthropic client instance. If not provided,
            a new client will be created using the ANTHROPIC_API_KEY env var.
        rpm: Rate limit in requests per minute.
        pace: Optional Pace instance for rate limiting.
        **kwargs: Additional arguments passed to the Anthropic client constructor.
    """

    client: AnthropicClient

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        api_key: str | None = None,
        rpm: int | None = None,
        pace: Any | None = None,
        **kwargs: dict,
    ):
        self_kwargs = {
            "rpm": rpm,
            "pace": pace,
            **kwargs,
        }
        self_kwargs["callback_class"] = AnthropicCallback

        if client is None:
            api_key = api_key or _get_env_api_key()
            client = anthropic.Anthropic(api_key=api_key, **kwargs)
            self_kwargs["client"] = AnthropicClient(client=client)
        else:
            if not isinstance(client, AnthropicClient):
                client = AnthropicClient(client=client)
            self_kwargs["client"] = client

        super().__init__(**self_kwargs)

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: core_endpoint.EndpointCallback | None,
    ) -> Any:
        callbacks = [self.global_callback]
        if callback is not None:
            callbacks.append(callback)

        cost_info = AnthropicCostComputer.handle_response(response)
        for cb in callbacks:
            cb.cost += _build_cost(cost_info)

        return response
