"""Feedback evaluation through API Route's OpenAI-compatible gateway."""

from __future__ import annotations

import os
import typing

from trulens.providers.openai import provider as openai_provider


class APIRoute(openai_provider.OpenAI):
    """Run text feedback functions using an API Route model.

    Reuses the OpenAI provider's capability probing and fallback behavior.
    Supported request options depend on the chosen model. Model IDs are sent
    unchanged; select an ID available to your API key via ``GET /v1/models``.
    This integration does not support moderation feedback or report API Route
    billing costs. Token usage remains available through the OpenAI endpoint.

    Args:
        model_engine: Model ID available to the key. Defaults to ``gpt-6.1-sol``.
        api_key: API Route key, or ``API_ROUTE_API_KEY`` from the environment.
        base_url: Explicit endpoint, then ``API_ROUTE_BASE_URL``, then
            ``https://global.api-route.com/v1``.
        **kwargs: Additional OpenAI endpoint/client options.

    Raises:
        ValueError: If no API Route key is supplied.
    """

    DEFAULT_MODEL_ENGINE: typing.ClassVar[str] = "gpt-6.1-sol"

    def __init__(
        self,
        *args,
        model_engine: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = os.environ.get("API_ROUTE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set API_ROUTE_API_KEY or pass api_key to the APIRoute provider."
            )
        if base_url is None:
            base_url = os.environ.get("API_ROUTE_BASE_URL") or (
                "https://global.api-route.com/v1"
            )
        super().__init__(
            *args,
            model_engine=model_engine or self.DEFAULT_MODEL_ENGINE,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @property
    def reports_costs(self) -> bool:
        """API Route prices are not the OpenAI endpoint's price estimates."""
        return False

    def _moderation(self, text: str):
        """Reject unsupported moderation feedback before making a request."""
        raise NotImplementedError(
            "Moderation feedback is not supported by the APIRoute provider."
        )


APIRoute.model_rebuild()
