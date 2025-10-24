import inspect
import json
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
        usage = response.usage_metadata

        endpoint = GoogleEndpoint()
        callback = GoogleCallback(endpoint=endpoint)

        model_name = response.model_version
        n_total_tokens = usage.total_token_count or 0
        n_prompt_tokens = usage.prompt_token_count or 0
        n_completion_tokens = usage.candidates_token_count or 0
        n_reasoning_tokens = usage.thoughts_token_count or 0

        return {
            SpanAttributes.COST.NUM_TOKENS: n_total_tokens,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: n_prompt_tokens,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: n_completion_tokens,
            SpanAttributes.COST.NUM_REASONING_TOKENS: n_reasoning_tokens,
            # TODO: Check the cost computation functionality
            SpanAttributes.COST.COST: callback._compute_cost(
                model_name, n_prompt_tokens, n_completion_tokens
            ),
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

        # Compute and set cost
        model_name = response_dict.get("model_version")
        n_prompt_tokens = usage.get("prompt_token_count", 0)
        n_completion_tokens = usage.get("candidates_token_count", 0)
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
        """Compute cost in USD based on model name and token counts.

        Args:
            model_name: Full model name from Google API (e.g., "gemini-2.5-flash-001")
            n_prompt_tokens: Number of input/prompt tokens
            n_completion_tokens: Number of output/completion tokens

        Returns:
            Cost in USD
        """
        try:
            if self._model_costs is None:
                # Load pricing configuration from JSON file
                # Reference: https://ai.google.dev/gemini-api/docs/pricing
                with open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "config/google_model_costs.json",
                    ),
                    "r",
                ) as f:
                    self._model_costs = json.load(f)

            logger.debug(f"Received model_version: {model_name}")
            if model_name and model_name in self._model_costs:
                if (
                    n_prompt_tokens > 2e5
                    and "input_large" in self._model_costs[model_name]
                ):
                    cost = (
                        self._model_costs[model_name]["input_large"]
                        * n_prompt_tokens
                        / 1e6
                    )
                else:
                    cost = (
                        self._model_costs[model_name]["input"]
                        * n_prompt_tokens
                        / 1e6
                    )
                if (
                    n_prompt_tokens > 2e5
                    and "output_large" in self._model_costs[model_name]
                ):
                    cost += (
                        self._model_costs[model_name]["output_large"]
                        * n_completion_tokens
                        / 1e6
                    )
                else:
                    cost += (
                        self._model_costs[model_name]["output"]
                        * n_completion_tokens
                        / 1e6
                    )
                return cost
            else:
                raise ValueError(
                    f"Model {model_name} not valid or not supported yet for cost estimation."
                )
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
