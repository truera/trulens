import json
import logging
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

import pydantic
from trulens.core.utils import pace as pace_utils
from trulens.feedback import llm_provider
from trulens.providers.anthropic import endpoint as anthropic_endpoint

logger = logging.getLogger(__name__)


class Anthropic(llm_provider.LLMProvider):
    """Out of the box feedback functions calling Anthropic Claude APIs.

    Additionally, all feedback functions listed in the base
    [LLMProvider class][trulens.feedback.LLMProvider] can be run with Anthropic.

    Create an Anthropic Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.anthropic import Anthropic
        anthropic_provider = Anthropic()
        ```

    Args:
        model_engine: The Anthropic model. Defaults to ``"claude-sonnet-4-6"``.
        api_key: Anthropic API key. If not provided, reads from
            ``ANTHROPIC_API_KEY`` env var.
        **kwargs: Additional arguments passed to the AnthropicEndpoint and
            ultimately to the ``anthropic.Anthropic`` client constructor.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "claude-sonnet-4-6"

    # Endpoint cannot presently be serialized but is constructed in __init__
    endpoint: anthropic_endpoint.AnthropicEndpoint = pydantic.Field(
        exclude=True
    )

    def __init__(
        self,
        *args,
        endpoint=None,
        api_key: Optional[str] = None,
        pace: Optional[pace_utils.Pace] = None,
        rpm: Optional[int] = None,
        model_engine: Optional[str] = None,
        **kwargs: dict,
    ):
        # NOTE: pydantic adds endpoint to the signature of this constructor
        # if we don't include it explicitly. We set it down below.

        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        self_kwargs: Dict[str, Any] = dict()
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine

        self_kwargs["endpoint"] = anthropic_endpoint.AnthropicEndpoint(
            *args,
            api_key=api_key,
            pace=pace,
            rpm=rpm,
            **kwargs,
        )

        super().__init__(**self_kwargs)

    @staticmethod
    def _extract_system_from_messages(
        messages: Sequence[Dict],
    ) -> tuple[str, List[Dict]]:
        """Extract system message(s) from OpenAI-format messages.

        Anthropic expects ``system`` as a top-level API parameter, not as a
        message role.

        Returns:
            (system_prompt, remaining_messages)
        """
        system_parts: List[str] = []
        remaining: List[Dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks — extract text parts
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            system_parts.append(block.get("text", ""))
            else:
                remaining.append(msg)
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        return system_prompt, remaining

    @staticmethod
    def _convert_messages_to_anthropic(
        messages: Sequence[Dict],
    ) -> List[Dict]:
        """Convert OpenAI-format messages to Anthropic Messages API format.

        Handles role translation and content structure conversion:
        - ``user`` → ``user`` with content blocks
        - ``assistant`` → ``assistant`` with content blocks
        - ``tool`` → ``user`` with tool_result content block
        - Merges consecutive same-role messages (Anthropic requires alternating)
        """
        anthropic_messages: List[Dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Determine Anthropic-compatible role
            if role == "system":
                continue  # Already extracted in _extract_system_from_messages
            elif role == "tool":
                # Anthropic: tool_result goes inside a user message
                anthropic_role = "user"
                tool_content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": (
                            content
                            if isinstance(content, str)
                            else json.dumps(content)
                        ),
                    }
                ]
            elif role == "assistant":
                anthropic_role = "assistant"
                if isinstance(content, str):
                    tool_content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    tool_content = content
                else:
                    tool_content = [
                        {"type": "text", "text": str(content)}
                    ]
            else:  # user or any other role → user
                anthropic_role = "user"
                if isinstance(content, str):
                    tool_content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    tool_content = content
                else:
                    tool_content = [
                        {"type": "text", "text": str(content)}
                    ]

            # Merge with previous message if same role (Anthropic requirement)
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == anthropic_role
            ):
                anthropic_messages[-1]["content"].extend(tool_content)
            else:
                anthropic_messages.append({
                    "role": anthropic_role,
                    "content": tool_content,
                })

        return anthropic_messages

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs,
    ) -> Optional[Union[str, pydantic.BaseModel]]:
        """Create a chat completion using the Anthropic Claude API.

        Args:
            prompt: Optional text prompt (used if messages not provided).
            messages: Optional sequence of OpenAI-format message dicts.
            response_format: Optional Pydantic model for structured output.
            **kwargs: Additional arguments (model, temperature, max_tokens, etc.)

        Returns:
            The model response text, or a parsed Pydantic model if
            ``response_format`` is provided.
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine

        # Build messages from prompt if no messages provided
        if messages is not None:
            input_messages = list(messages)
        elif prompt is not None:
            input_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        # Extract system prompt and convert messages
        system_prompt, non_system_msgs = self._extract_system_from_messages(
            input_messages
        )
        anthropic_messages = self._convert_messages_to_anthropic(
            non_system_msgs
        )

        # Build API parameters
        api_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model"),
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": anthropic_messages,
        }

        if system_prompt:
            api_kwargs["system"] = system_prompt

        # Handle temperature — non-reasoning Anthropic models support it
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs.pop("temperature")
        elif not self._is_reasoning_model():
            api_kwargs["temperature"] = 0.0

        # Handle structured output via tool_use
        if response_format is not None:
            try:
                schema = response_format.model_json_schema()
                api_kwargs["tools"] = [
                    {
                        "name": "output_format",
                        "description": (
                            "Respond with the required structured output."
                        ),
                        "input_schema": schema,
                    }
                ]
                api_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": "output_format",
                }
            except Exception as e:
                logger.debug(
                    "Failed to generate tool schema for response_format "
                    f"{response_format.__name__}: {e}. "
                    "Falling back to text output.",
                    exc_info=True,
                )

        # Merge any remaining kwargs
        api_kwargs.update(kwargs)

        # Call Anthropic API
        response = self.endpoint.run_in_pace(
            func=self.endpoint.client.client.messages.create,
            **api_kwargs,
        )

        # Handle structured output response
        if response_format is not None:
            for content_block in getattr(response, "content", []) or []:
                if getattr(content_block, "type", None) == "tool_use":
                    tool_input = getattr(content_block, "input", {}) or {}
                    try:
                        return response_format.model_validate(tool_input)
                    except Exception as e:
                        logger.debug(
                            "Failed to parse tool_use output as "
                            f"{response_format.__name__}: {e}",
                            exc_info=True,
                        )
                        # Fall through to text extraction

        # Extract text content
        text_parts: List[str] = []
        for content_block in getattr(response, "content", []) or []:
            if getattr(content_block, "type", None) == "text":
                text = getattr(content_block, "text", "")
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts) if text_parts else ""
