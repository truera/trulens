"""Endpoint and usage tracking for the TypeSafe SDK."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import pydantic
from trulens.core.feedback import endpoint as core_endpoint


class TypeSafeCallback(core_endpoint.EndpointCallback):
    """Track Jev classification requests and token usage."""

    def handle_classification(self, response: Any) -> None:
        """Record one successful typed-decision request."""
        super().handle_classification(response)
        self.cost.n_successful_requests += 1
        answers = getattr(response, "answers", {})
        self.cost.n_classes += len(answers)
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            self.cost.n_prompt_tokens += input_tokens
            self.cost.n_completion_tokens += output_tokens
            self.cost.n_tokens += input_tokens + output_tokens


class TypeSafeEndpoint(core_endpoint.Endpoint):
    """Paced, retrying endpoint for TypeSafe System One requests."""

    callback_class: ClassVar[type[TypeSafeCallback]] = TypeSafeCallback
    client: Any = pydantic.Field(exclude=True)

    def __init__(
        self,
        *,
        model_engine: str,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        rpm: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            client_kwargs = {
                "model": model_engine,
                "timeout": timeout,
            }
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            client = TypeSafeClient(**client_kwargs)
        super().__init__(
            name="typesafe",
            rpm=rpm,
            callback_class=self.callback_class,
            client=client,
            **kwargs,
        )

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: str,
    ) -> Any:
        """Run a System One request under endpoint pacing and retries."""
        response = self.run_in_pace(
            self.client.system_one,
            state=state,
            questions=questions,
            model=model,
        )
        self.global_callback.handle_classification(response)
        return response
