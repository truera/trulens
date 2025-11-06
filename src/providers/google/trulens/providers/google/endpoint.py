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

from litellm import model_cost
from trulens.core.feedback import endpoint as core_endpoint
from trulens.otel.semconv.trace import SpanAttributes

from google import genai
from google.auth.credentials import Credentials
from google.genai import Client
from google.genai.types import GenerateContentResponse

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


def _get_env_api_key() -> Optional[str]:
    """Gets the API key from environment variables, prioritizing GOOGLE_API_KEY.
    Returns:
        The API key string if found, otherwise None. Empty string is considered
        invalid.
    """
    env_google_api_key = os.environ.get("GOOGLE_API_KEY", None)
    env_gemini_api_key = os.environ.get("GEMINI_API_KEY", None)
    if env_google_api_key and env_gemini_api_key:
        logger.warning(
            "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
        )

    return env_google_api_key or env_gemini_api_key or None


class GoogleCostComputer:
    @staticmethod
    def handle_response(response: Any) -> Dict[str, Any]:
        """Process Google API response and extract cost/usage metadata.

        Args:
            response: GenerateContentResponse object from Google API

        Returns:
            Dictionary with cost and usage attributes for OpenTelemetry span
        """
        usage = response.usage_metadata

        endpoint = GoogleEndpoint()
        callback = GoogleCallback(endpoint=endpoint)

        model_name = response.model_version
        n_total_tokens = usage.total_token_count or 0
        n_prompt_tokens = usage.prompt_token_count or 0
        n_completion_tokens = usage.candidates_token_count or 0
        n_reasoning_tokens = usage.thoughts_token_count or 0
        calculated_cost = callback._compute_cost(
            model_name, n_prompt_tokens, n_completion_tokens
        )

        return {
            SpanAttributes.COST.NUM_TOKENS: n_total_tokens,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: n_prompt_tokens,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: n_completion_tokens,
            SpanAttributes.COST.NUM_REASONING_TOKENS: n_reasoning_tokens,
            SpanAttributes.COST.COST: calculated_cost,
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.MODEL: model_name,
        }


class GoogleCallback(core_endpoint.EndpointCallback):
    _FIELDS_MAP: ClassVar[List[Tuple[str, str]]] = [
        ("cost", "total_cost"),
        ("n_tokens", "total_tokens"),
        ("n_successful_requests", "successful_requests"),
        ("n_prompt_tokens", "prompt_tokens"),
        ("n_completion_tokens", "completion_tokens"),
    ]

    _model_costs: Optional[dict] = None

    def handle_generation(self, response: Any):
        """Get the usage information from GoogleGenAI LLM function response's usage_metadata field."""
        response_dict = response
        if isinstance(response, GenerateContentResponse):
            response_dict = response.to_json_dict()

        usage = response_dict.get("usage_metadata")
        super().handle_generation(response_dict)
        self.cost.n_successful_requests += 1

        for cost_field, google_field in [
            ("n_tokens", "total_token_count"),
            ("n_prompt_tokens", "prompt_token_count"),
            ("n_completion_tokens", "candidates_token_count"),
            ("n_reasoning_tokens", "thoughts_token_count"),
        ]:
            setattr(
                self.cost,
                cost_field,
                getattr(self.cost, cost_field, 0) + usage.get(google_field, 0),
            )

        model_name = response_dict.get("model_version")
        n_prompt_tokens = usage.get("prompt_token_count", 0)
        n_completion_tokens = usage.get("candidates_token_count", 0)

        # Try LiteLLM first
        calculated_cost = self._compute_cost(
            model_name, n_prompt_tokens, n_completion_tokens
        )

        setattr(
            self.cost, "cost", getattr(self.cost, "cost", 0) + calculated_cost
        )
        setattr(self.cost, "cost_currency", "USD")

    def _compute_cost(
        self, model_name: str, n_prompt_tokens: int, n_completion_tokens: int
    ) -> float:
        """Compute cost in USD based on model name and token counts using LiteLLM community-maintained pricing list.
        Reference: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

        Args:
            model_name: Full model name from Google API (e.g., "gemini-2.5-flash-001")
            n_prompt_tokens: Number of input/prompt tokens
            n_completion_tokens: Number of output/completion tokens

        Returns:
            Cost in USD
        """
        try:
            self._model_costs = model_cost

            if not model_name:
                logger.warning("No model name provided for cost calculation")
                return 0.0

            logger.debug(f"Received model_version: {model_name}")

            if model_name in self._model_costs:
                pricing = self._model_costs[model_name]

                if n_prompt_tokens > pricing.get(
                    "max_input_tokens"
                ) or n_completion_tokens > pricing.get("max_output_tokens"):
                    logger.warning(
                        f"Model {model_name} has exceeded the maximum input or output tokens. Skipping cost calculation."
                    )
                    return 0.0

                # Determine input pricing based on prompt size (<=200K vs >200K tokens)
                if n_prompt_tokens > 200000:
                    input_price = pricing.get(
                        "input_cost_per_token_above_200k_tokens",
                        pricing.get("input_cost_per_token", 0),
                    )
                else:
                    input_price = pricing.get("input_cost_per_token", 0)

                # Determine output pricing based on prompt size (<=200K vs >200K tokens)
                if n_prompt_tokens > 200000:
                    output_price = pricing.get(
                        "output_cost_per_token_above_200k_tokens",
                        pricing.get("output_cost_per_token", 0),
                    )
                else:
                    output_price = pricing.get("output_cost_per_token", 0)

                # Calculate total cost
                cost = (
                    n_prompt_tokens * input_price
                    + n_completion_tokens * output_price
                )
                logger.debug(
                    f"JSON pricing cost calculated: ${cost:.6f} for {model_name} "
                )
                return cost

            # Model not found in pricing config
            logger.warning(
                f"Model {model_name} not found in pricing configuration. "
                f"Cost tracking will be incomplete. Available models: {list(self._model_costs.keys())}"
            )
            return 0.0

        except Exception as e:
            logger.error(
                f"Error occurred while computing cost for model {model_name}: {e}"
            )
            return 0.0


class GoogleEndpoint(core_endpoint.Endpoint):
    client: Optional["Client"] = None
    vertexai: Optional[bool] = None
    api_key: Optional[str] = None
    credentials: Optional["Credentials"] = None
    project: Optional[str] = None
    location: Optional[str] = None

    def __init__(
        self,
        client: Optional["Client"] = None,
        vertexai: Optional[bool] = None,
        api_key: Optional[str] = None,
        credentials: Optional["Credentials"] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: dict,
    ):
        actual_client_instance = client
        if not actual_client_instance:
            if vertexai:
                actual_client_instance = genai.Client(
                    vertexai=vertexai,
                    credentials=credentials,
                    project=project,
                    location=location,
                )
            else:
                actual_client_instance = genai.Client(
                    api_key=api_key or _get_env_api_key()
                )

        kwargs_for_super = {
            # These are the fields declared in GoogleEndpoint
            "client": actual_client_instance,
            "vertexai": vertexai,
            "api_key": api_key,
            "credentials": credentials,
            "project": project,
            "location": location,
            # This is a kwarg your Endpoint base class expects,
            # and it's also set for the parent.
            "callback_class": GoogleCallback,
            **kwargs,  # Pass through any other arbitrary kwargs
        }

        super().__init__(**kwargs_for_super)

    def handle_wrapped_call(
        self,
        func: Callable[..., Any],
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        try:
            if isinstance(response, GenerateContentResponse):
                response_dict = response.to_json_dict()
        except Exception as e:
            logger.error(f"Error occurred while parsing response: {e}")
            raise e

        if (
            isinstance(response_dict, dict)
            and "usage_metadata" in response_dict
        ):
            candidate = response_dict["candidates"][0]
            if (
                "finish_reason" in candidate
                and candidate["finish_reason"] == "STOP"
            ):
                self.global_callback.handle_generation(response=response_dict)

                if callback is not None:
                    callback.handle_generation(response=response_dict)
        else:
            logger.warning(
                "Unrecognized Google content response format. It did not have usage information:\n%s",
                pp.pformat(response_dict),
            )

        return response
