import logging
import os
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx
from langchain_core.outputs import LLMResult
import pydantic
from trulens.providers.openai import provider as openai_provider

logger = logging.getLogger(__name__)


class Ollama(openai_provider.OpenAI):
    """Out of the box feedback functions calling models served locally (or
    remotely) by [Ollama](https://ollama.com/).

    Unlike routing through [LiteLLM][trulens.providers.litellm.LiteLLM] or
    the OpenAI-compatible `/v1` API, this provider talks to Ollama's
    **native** `/api/chat` endpoint directly. That's required to actually
    support Ollama-specific functionality: per-request `options`
    (Modelfile-style parameters like `num_ctx`, `mirostat`,
    `repeat_penalty`, ...), `keep_alive` control, and schema-constrained
    structured output. It also exposes model management (pulling/listing
    models). None of this is reachable through Ollama's OpenAI-compatible
    surface, which silently ignores those fields.

    Construction still reuses [OpenAI][trulens.providers.openai.OpenAI]'s
    `__init__` (so request pacing, moderation endpoints, and serialization
    all keep working the same way), preconfigured to point at a local Ollama
    server with no API key required.

    !!! warning
        _Ollama_ does not support the _OpenAI_ moderation endpoint.

    Create an Ollama Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.ollama import Ollama
        ollama_provider = Ollama(model_engine="llama3.2")
        ```

        By default, the provider connects to `http://localhost:11434`. To
        connect to a remote Ollama instance, either set the `OLLAMA_BASE_URL`
        environment variable or pass `base_url` explicitly:

        ```python
        from trulens.providers.ollama import Ollama

        provider = Ollama(
            model_engine="llama3.2",
            base_url="http://my-ollama-host:11434/v1",
        )
        ```

        Pull a model onto the server and list what's available:

        ```python
        provider.pull_model("llama3.2")
        provider.list_models()  # ["llama3.2:latest", ...]
        ```

        Control Ollama-specific generation options and how long the model
        stays resident in memory:

        ```python
        provider = Ollama(
            model_engine="llama3.2",
            options={"num_ctx": 8192, "repeat_penalty": 1.2},
            keep_alive="10m",
        )
        ```

    Args:
        model_engine: The Ollama model to use, e.g. `"llama3.2"`. Must
            already be pulled (`ollama pull llama3.2`, or
            [`pull_model`][trulens.providers.ollama.provider.Ollama.pull_model])
            on the target server. Defaults to `llama3.2`.

        base_url: The base URL of the Ollama server. Defaults to the
            `OLLAMA_BASE_URL` environment variable if set, otherwise
            `http://localhost:11434/v1`. A trailing `/v1` (OpenAI-compat
            convention) is accepted and stripped when calling Ollama's
            native API.

        options: Ollama-specific model options (e.g. `num_ctx`,
            `repeat_penalty`, `mirostat`, `top_k`, `temperature`, ...) sent
            with every request. See the
            [Modelfile valid parameters](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
            for the full list Ollama understands. Per-call `options` passed
            to a feedback function are merged on top of these.

        keep_alive: How long the model stays loaded in memory after a
            request, e.g. `"5m"`, or `-1` to keep it loaded indefinitely.
            Sent with every request.

        **kwargs: Additional arguments to pass to the
            [OpenAIEndpoint][trulens.providers.openai.endpoint.OpenAIEndpoint]
            which are then passed to
            [OpenAIClient][trulens.providers.openai.endpoint.OpenAIClient]
            and finally to the OpenAI client (used for request pacing and
            the -- unsupported by Ollama -- moderation endpoints only;
            chat completions bypass it entirely).
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "llama3.2"

    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434/v1"
    """Default base URL of a local Ollama server."""

    options: Optional[Dict[str, Any]] = None
    """Ollama-specific model options (e.g. `num_ctx`, `temperature`,
    `repeat_penalty`, `mirostat`, ...) sent with every request."""

    keep_alive: Optional[Union[str, int]] = None
    """How long the model stays loaded in memory after a request, e.g.
    `"5m"`, or `-1` to keep it loaded indefinitely. Sent with every
    request."""

    def __init__(
        self,
        *args,
        model_engine: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[str, int]] = None,
        **kwargs: dict,
    ):
        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)

        if api_key is None:
            # Ollama does not require an API key, but the underlying OpenAI
            # client requires the value to be a non-empty string.
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

        super().__init__(
            *args,
            model_engine=model_engine,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

        # Set after `super().__init__()` since these are Ollama-only fields
        # that the OpenAI client constructor does not accept.
        self.options = options
        self.keep_alive = keep_alive

    def _native_base_url(self) -> str:
        """The Ollama server's native API root (without the `/v1` suffix
        used for OpenAI-compatible requests), e.g.
        `http://localhost:11434`."""
        base = str(self.endpoint.client.client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def list_models(self) -> List[str]:
        """List the names of models currently available on the Ollama
        server.

        Example:
            ```python
            from trulens.providers.ollama import Ollama
            provider = Ollama()
            provider.list_models()  # e.g. ["llama3.2:latest"]
            ```

        Returns:
            List of model names, as reported by the server.
        """
        response = httpx.get(
            f"{self._native_base_url()}/api/tags", timeout=30.0
        )
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def pull_model(self, name: Optional[str] = None) -> None:
        """Pull (download) a model onto the Ollama server, blocking until
        the pull completes.

        Example:
            ```python
            from trulens.providers.ollama import Ollama
            provider = Ollama(model_engine="llama3.2")
            provider.pull_model()  # pulls "llama3.2"
            ```

        Args:
            name: Model to pull, e.g. `"llama3.2"`. Defaults to this
                provider's configured `model_engine`.
        """
        target = name or self.model_engine
        response = httpx.post(
            f"{self._native_base_url()}/api/pull",
            json={"name": target, "stream": False},
            timeout=httpx.Timeout(10.0, read=None),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"Failed to pull model {target!r}: {payload}")

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        *,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> Optional[str]:
        if messages is not None:
            input_messages = list(messages)
        elif prompt is not None:
            input_messages = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        model_name = kwargs.pop("model", self.model_engine)

        # Ollama exposes generation parameters (temperature, seed, top_p,
        # ...) nested under `options`, not as top-level request fields like
        # the OpenAI API.
        effective_options: Dict[str, Any] = dict(self.options or {})
        if options:
            effective_options.update(options)
        for key in ("temperature", "seed", "top_p"):
            if key in kwargs:
                effective_options[key] = kwargs.pop(key)

        # Ollama has no native `reasoning_effort` control; local reasoning
        # models (e.g. deepseek-r1) surface thinking in the response text
        # instead, so this knob is a no-op for Ollama.
        kwargs.pop("reasoning_effort", None)

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": input_messages,
            "stream": False,
        }
        if effective_options:
            body["options"] = effective_options

        effective_keep_alive = (
            keep_alive if keep_alive is not None else self.keep_alive
        )
        if effective_keep_alive is not None:
            body["keep_alive"] = effective_keep_alive

        if response_format is not None:
            body["format"] = response_format.model_json_schema()

        response = self.endpoint.run_in_pace(
            func=httpx.post,
            url=f"{self._native_base_url()}/api/chat",
            json=body,
            timeout=httpx.Timeout(10.0, read=300.0),
        )
        response.raise_for_status()
        payload = response.json()

        self._record_native_usage(payload, model_name)

        return payload.get("message", {}).get("content", "")

    def _record_native_usage(
        self, payload: Dict[str, Any], model_name: str
    ) -> None:
        """Feed Ollama's native token counts into the same cost-tracking
        pipeline the OpenAI client would normally populate automatically,
        since native `/api/chat` calls bypass that instrumentation."""
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        if prompt_tokens is None and completion_tokens is None:
            return

        usage = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }
        llm_result = LLMResult(
            generations=[[]],
            llm_output=dict(token_usage=usage, model_name=model_name),
            run=None,
        )
        self.endpoint.global_callback.handle_generation(response=llm_result)
