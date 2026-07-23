import inspect
import logging
import os
import pprint
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)

import pydantic
from trulens.core.feedback import endpoint as core_endpoint
from trulens.otel.semconv.trace import SpanAttributes

import anthropic
from anthropic.types import Message as AnthropicMessage

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

# Anthropic model pricing per 1M tokens (input, output), in USD
# Updated June 2026
ANTHROPIC_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (0.80, 4.0),
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
}
DEFAULT_PRICE_PER_1M_INPUT = 3.0
DEFAULT_PRICE_PER_1M_OUTPUT = 15.0


def _get_env_api_key() -> Optional[str]:
    """Gets the API key from ANTHROPIC_API_KEY environment variable."""
    return os.environ.get("ANTHROPIC_API_KEY", None)


def _get_model_pricing(model_name: str) -> Tuple[float, float]:
    """Get (input_price_per_1M, output_price_per_1M) for a model.

    Performs prefix matching to handle model version suffixes.
    """
    if not model_name:
        return (DEFAULT_PRICE_PER_1M_INPUT, DEFAULT_PRICE_PER_1M_OUTPUT)
    for prefix, prices in ANTHROPIC_PRICING.items():
        if model_name.startswith(prefix):
            return prices
    return (DEFAULT_PRICE_PER_1M_INPUT, DEFAULT_PRICE_PER_1M_OUTPUT)


class AnthropicCostComputer:
    """Computes cost and token usage from Anthropic API responses."""

    @staticmethod
    def handle_response(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        model_name = getattr(response, "model", "") or ""

        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        # Cache tokens include previously-seen content (not separately billed)
        cache_read_input_tokens = (
            getattr(usage, "cache_read_input_tokens", 0) if usage else 0
        )
        cache_creation_input_tokens = (
            getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
        )

        input_price, output_price = _get_model_pricing(model_name)
        cost = (input_tokens / 1_000_000.0) * input_price + (
            output_tokens / 1_000_000.0
        ) * output_price

        return {
            SpanAttributes.COST.COST: round(cost, 8),
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.NUM_TOKENS: total_tokens,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: input_tokens,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: output_tokens,
            SpanAttributes.COST.NUM_REASONING_TOKENS: 0,
            SpanAttributes.COST.MODEL: model_name,
        }


class AnthropicCallback(core_endpoint.EndpointCallback):
    """Callback for Anthropic endpoint instrumentation."""

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)
        cost_info = AnthropicCostComputer.handle_response(response)

        addl_cost = core_endpoint.Cost(
            cost=cost_info.get(SpanAttributes.COST.COST, 0.0),
            currency=cost_info.get(SpanAttributes.COST.CURRENCY, "USD"),
            n_tokens=cost_info.get(SpanAttributes.COST.NUM_TOKENS, 0),
            n_prompt_tokens=cost_info.get(
                SpanAttributes.COST.NUM_PROMPT_TOKENS, 0
            ),
            n_completion_tokens=cost_info.get(
                SpanAttributes.COST.NUM_COMPLETION_TOKENS, 0
            ),
            n_reasoning_tokens=cost_info.get(
                SpanAttributes.COST.NUM_REASONING_TOKENS, 0
            ),
        )
        self.cost += addl_cost


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
        client: Optional[anthropic.Anthropic] = None,
        api_key: Optional[str] = None,
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
        client: Optional[anthropic.Anthropic] = None,
        api_key: Optional[str] = None,
        rpm: Optional[int] = None,
        pace: Optional[Any] = None,
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
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        model_name = ""
        if "model" in bindings.kwargs:
            model_name = bindings.kwargs["model"]
        elif "model" in bindings.arguments:
            model_name = bindings.arguments["model"]

        callbacks = [self.global_callback]
        if callback is not None:
            callbacks.append(callback)

        cost_info = AnthropicCostComputer.handle_response(response)
        for cb in callbacks:
            addl_cost = core_endpoint.Cost(
                cost=cost_info.get(SpanAttributes.COST.COST, 0.0),
                currency=cost_info.get(SpanAttributes.COST.CURRENCY, "USD"),
                n_tokens=cost_info.get(SpanAttributes.COST.NUM_TOKENS, 0),
                n_prompt_tokens=cost_info.get(
                    SpanAttributes.COST.NUM_PROMPT_TOKENS, 0
                ),
                n_completion_tokens=cost_info.get(
                    SpanAttributes.COST.NUM_COMPLETION_TOKENS, 0
                ),
                n_reasoning_tokens=cost_info.get(
                    SpanAttributes.COST.NUM_REASONING_TOKENS, 0
                ),
            )
            cb.cost += addl_cost

        return response
