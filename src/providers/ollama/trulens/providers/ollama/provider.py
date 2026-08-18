import logging
import os
from typing import ClassVar, Optional

from trulens.providers.openai import provider as openai_provider

logger = logging.getLogger(__name__)


class Ollama(openai_provider.OpenAI):
    """Out of the box feedback functions calling models served locally (or
    remotely) by [Ollama](https://ollama.com/).

    Ollama exposes an OpenAI-compatible `/v1` API, so this provider is a thin
    wrapper around [OpenAI][trulens.providers.openai.OpenAI] that is
    preconfigured to talk to an Ollama server without requiring an API key.

    !!! warning
        _Ollama_ does not support the _OpenAI_ moderation endpoint.

    Create an Ollama Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.ollama import Ollama
        ollama_provider = Ollama(model_engine="llama3.2")
        ```

        By default, the provider connects to `http://localhost:11434/v1`. To
        connect to a remote Ollama instance, either set the `OLLAMA_BASE_URL`
        environment variable or pass `base_url` explicitly:

        ```python
        from trulens.providers.ollama import Ollama

        provider = Ollama(
            model_engine="llama3.2",
            base_url="http://my-ollama-host:11434/v1",
        )
        ```

    Args:
        model_engine: The Ollama model to use, e.g. `"llama3.2"`. Must
            already be pulled (`ollama pull llama3.2`) on the target server.
            Defaults to `llama3.2`.

        base_url: The base URL of the Ollama server's OpenAI-compatible API.
            Defaults to the `OLLAMA_BASE_URL` environment variable if set,
            otherwise `http://localhost:11434/v1`.

        **kwargs: Additional arguments to pass to the
            [OpenAIEndpoint][trulens.providers.openai.endpoint.OpenAIEndpoint]
            which are then passed to
            [OpenAIClient][trulens.providers.openai.endpoint.OpenAIClient]
            and finally to the OpenAI client.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "llama3.2"

    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434/v1"
    """Default base URL of a local Ollama server's OpenAI-compatible API."""

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
