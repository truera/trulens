import logging
import os
from typing import ClassVar, Optional

from trulens.providers.openai import provider as openai_provider

logger = logging.getLogger(__name__)


class OrcaRouter(openai_provider.OpenAI):
    """Out of the box feedback functions calling models through the
    [OrcaRouter](https://www.orcarouter.ai) gateway.

    OrcaRouter is an OpenAI-compatible gateway that exposes a provider-prefixed
    model namespace (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4.6`,
    `google/gemini-2.5-pro`, `orcarouter/auto`) behind a single base URL, so a
    single feedback provider can evaluate against many models. Other
    OpenAI-compatible gateways are reachable through the OpenAI provider's
    `base_url` forwarding; this provider instead wires the gateway's default
    endpoint and API key (`ORCAROUTER_API_KEY`) directly, so no plumbing is
    needed.

    Because OrcaRouter is API-compatible with OpenAI, the OpenAI
    implementation is reused as-is: capability probing, structured outputs and
    the Responses API all keep working.

    !!! warning
        _OrcaRouter_ does not currently expose the _OpenAI_ moderation
        endpoint (e.g. `text-moderation-stable`), so the `moderation_*`
        feedback functions will not work through this provider.

    Create an OrcaRouter Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.orcarouter import OrcaRouter

        # Uses ORCAROUTER_API_KEY and the default model openai/gpt-4o-mini.
        provider = OrcaRouter()
        ```

        Pick a different model from the gateway catalog by its provider-prefixed
        ID:

        ```python
        provider = OrcaRouter(model_engine="anthropic/claude-sonnet-4.6")
        ```

    Args:
        model_engine: The gateway model ID to use, e.g. `openai/gpt-4o-mini`.
            Defaults to `openai/gpt-4o-mini`.

        base_url: The OrcaRouter OpenAI-compatible base URL. Defaults to the
            `ORCAROUTER_BASE_URL` environment variable if set, otherwise
            `https://api.orcarouter.ai/v1`.

        api_key: The OrcaRouter API key. Defaults to the `ORCAROUTER_API_KEY`
            environment variable. Must be set either here or in the
            environment.

        **kwargs: Additional arguments to pass to the
            [OpenAIEndpoint][trulens.providers.openai.endpoint.OpenAIEndpoint]
            which are then passed to
            [OpenAIClient][trulens.providers.openai.endpoint.OpenAIClient]
            and finally to the OpenAI client (for example `api_key`,
            `base_url`).
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "openai/gpt-4o-mini"

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.orcarouter.ai/v1"
    """Default OrcaRouter OpenAI-compatible base URL."""

    def __init__(
        self,
        *args,
        model_engine: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: dict,
    ):
        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        if base_url is None:
            base_url = os.environ.get(
                "ORCAROUTER_BASE_URL", self.DEFAULT_BASE_URL
            )

        if api_key is None:
            api_key = os.environ.get("ORCAROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "ORCAROUTER_API_KEY must be set in the environment or "
                    "passed as `api_key` to the OrcaRouter provider."
                )

        super().__init__(
            *args,
            model_engine=model_engine,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
