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
        return {
            SpanAttributes.COST.NUM_TOKENS: usage.total_token_count or 0,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: usage.prompt_token_count
            or 0,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: usage.candidates_token_count
            or 0,
            SpanAttributes.COST.NUM_REASONING_TOKENS: usage.thoughts_token_count
            or 0,
            # TODO: Check the cost computation functionality
            # SpanAttributes.COST.COST: completion_cost(response),
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.MODEL: response.model_version,
        }


class GoogleCallback(core_endpoint.EndpointCallback):
    _FIELDS_MAP: ClassVar[List[Tuple[str, str]]] = [
        ("cost", "total_cost"),
        ("n_tokens", "total_tokens"),
        ("n_successful_requests", "successful_requests"),
        ("n_prompt_tokens", "prompt_tokens"),
        ("n_completion_tokens", "completion_tokens"),
    ]

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

        # TODO: missing code for cost calculation


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
