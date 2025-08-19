import logging
from typing import ClassVar, Dict, Optional, Sequence, Type, Union

import pydantic
from trulens.feedback import llm_provider
from trulens.providers.google import endpoint as google_endpoint

from google.auth.credentials import Credentials
from google.genai import Client
from google.genai.types import GenerateContentConfig

logger = logging.getLogger(__name__)


class Google(llm_provider.LLMProvider):
    """Google provides access to Google's generative models via the Gemini Developer API
    or Vertex AI, depending on the configuration.

    For more details, see the official Gemini documentation.

    Examples:

    === "Connecting with a Gemini Developer API client"
        ```python
        from google import genai
        from trulens.providers.google import Google

        google_client = genai.Client(api_key="GOOGLE_API_KEY")
        provider = Google(client=google_client)
        ```

    === "Connecting with a Vertex AI client"
        ```python
        from google import genai
        from trulens.providers.google import Google

        PROJECT_ID = "your_project_id"
        LOCATION = "us-central1"

        vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        provider = Google(client=vertex_client)
        ```

    === "Using only an API key (Gemini Developer API)"
        ```python
        from trulens.providers.google import Google

        provider = Google(api_key="GOOGLE_API_KEY")
        ```

    === "Using Vertex AI configuration directly"
        ```python
        from trulens.providers.google import Google

        PROJECT_ID = "your_project_id"
        LOCATION = "us-central1"

        provider = Google(vertexai=True, project=PROJECT_ID, location=LOCATION)
        ```

    Args:
        model_engine: Model engine to use. Defaults to `"gemini-2.5-flash"`.
        api_key: API key for authenticating with the Gemini Developer API. If not provided,
        the key will be read from the environment variable `GOOGLE_API_KEY` or `GEMINI_API_KEY`, if available.
        vertexai: Whether to use Vertex AI endpoints. Set to `True` to use Vertex AI instead of the Gemini Developer API. Defaults to `False`.
        credentials: Credentials to authenticate with Vertex AI. If not provided, default application credentials are used.
        project: Google Cloud project ID used for billing and quota when using Vertex AI. Can be set via environment variables.
        location: Region to send Vertex AI API requests to (e.g., `"us-central1"`). Can also be set via environment variables.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "gemini-2.5-flash"

    def __init__(
        self,
        endpoint=None,
        client: Optional["Client"] = None,
        vertexai: Optional[bool] = None,
        api_key: Optional[str] = None,
        credentials: Optional["Credentials"] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model_engine: Optional[str] = None,
        **kwargs: Dict,
    ):
        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine

        endpoint = google_endpoint.GoogleEndpoint(
            client=client,
            vertexai=vertexai,
            api_key=api_key,
            credentials=credentials,
            project=project,
            location=location,
            **kwargs,
        )
        self_kwargs["endpoint"] = endpoint
        super().__init__(**self_kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs,
    ) -> Optional[Union[str, pydantic.BaseModel]]:
        contents = []
        system_instruction = ""
        if messages is not None:
            for message in messages:
                if message["role"] == "system":
                    system_instruction = message["content"]
                elif message["role"] == "user":
                    # TODO: Add multi-modal (text + image) handling here for Google models
                    contents.append({
                        "parts": [{"text": message["content"]}],
                        "role": "user",
                    })
                else:
                    logger.warning(
                        f"Ignoring role '{message['role']}' — only 'system' and 'user' are supported."
                    )
        elif prompt is not None:
            contents.append({
                "parts": [{"text": prompt}],
                "role": "user",
            })
        else:
            raise ValueError("`prompt` or `messages` must be specified.")
        config_kwargs = dict(**kwargs)

        # Ensure seed is set if response_format is not used
        if response_format is None and "seed" not in config_kwargs:
            config_kwargs["seed"] = 123

        # Add optional fields based on conditions
        if response_format is not None and self._structured_output_supported():
            config_kwargs.update({
                "response_mime_type": "application/json",
                "response_schema": response_format,
            })

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.endpoint.client.models.generate_content(
            model=self.model_engine,
            contents=contents,
            config=GenerateContentConfig(**config_kwargs),
        )

        if response_format:
            return response.parsed
        return response.text

    def _structured_output_supported(self) -> bool:
        """Whether the provider supports structured output.
        For more details: https://ai.google.dev/gemini-api/docs/models
        """
        # Models with only output audio do not support structured output
        # generation (very logical)
        audio_only_output_models = [
            "gemini-2.5-pro-preview-tts",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-flash-preview-native-audio-dialog",
            "gemini-2.5-flash-exp-native-audio-thinking-dialog",
        ]
        if self.model_engine in audio_only_output_models:
            return False
        return True
