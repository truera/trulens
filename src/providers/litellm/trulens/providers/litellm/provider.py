import json
import logging
import os
import re
from typing import ClassVar, Dict, Optional, Sequence, Type

import pydantic
from pydantic import BaseModel
from trulens.core.feedback import endpoint as core_endpoint
from trulens.feedback import llm_provider as llm_provider
from trulens.providers.litellm import endpoint as litellm_endpoint

from litellm import completion
from litellm import get_supported_openai_params

logger = logging.getLogger(__name__)


class LiteLLM(llm_provider.LLMProvider):
    """Out of the box feedback functions calling LiteLLM API.

    Create an LiteLLM Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.litellm import LiteLLM
        litellm_provider = LiteLLM()
        ```
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "gpt-3.5-turbo"

    model_engine: str
    """The LiteLLM completion model. Defaults to `gpt-3.5-turbo`."""

    completion_args: Dict[str, str] = pydantic.Field(default_factory=dict)
    """Additional arguments to pass to the `litellm.completion` as needed for chosen api."""

    endpoint: core_endpoint.Endpoint

    def __init__(
        self,
        model_engine: Optional[str] = None,
        completion_kwargs: Optional[Dict] = None,
        endpoint: Optional[core_endpoint.Endpoint] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        from litellm.utils import get_llm_provider

        litellm_provider = get_llm_provider(model_engine)[1]

        if completion_kwargs is None:
            completion_kwargs = {}

        if model_engine.startswith("azure/") and (
            "api_base" not in completion_kwargs
            and not os.getenv("AZURE_API_BASE")
        ):
            raise ValueError(
                "Azure model engine requires 'api_base' parameter to litellm completions. "
                "Provide it to LiteLLM provider in the 'completion_kwargs' parameter:"
                """
                ```python
                provider = LiteLLM(
                    "azure/your_deployment_name",
                    completion_kwargs={
                        "api_base": "https://yourendpoint.openai.azure.com/"
                    }
                )
                ```
                """
            )

        self_kwargs: Dict[str, object] = {}
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine
        self_kwargs["litellm_provider"] = litellm_provider
        # store completion kwargs dict on the provider
        self_kwargs["completion_args"] = dict(completion_kwargs)
        self_kwargs["endpoint"] = litellm_endpoint.LiteLLMEndpoint(
            litellm_provider=litellm_provider, **kwargs
        )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # Prefer caching keyed by provider+model to avoid cross-provider collisions
    def _capabilities_key(self) -> str:  # type: ignore[override]
        provider = getattr(self, "litellm_provider", "") or "unknown"
        return f"{provider}:{self.model_engine}"

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        def _postprocess_content(text: str) -> str:
            if not isinstance(text, str):
                return text
            if self._is_reasoning_model():
                return self._sanitize_reasoning_output(text)
            return text

        completion_args = dict(kwargs)
        completion_args["model"] = self.model_engine
        completion_args.update(self.completion_args)

        if messages is not None:
            completion_args["messages"] = messages
        elif prompt is not None:
            completion_args["messages"] = [
                {"role": "system", "content": prompt}
            ]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        if response_format is not None:
            completion_args["response_format"] = response_format

        if "/" in self.model_engine:
            # if we have model provider and model name, verify params
            _model_provider, _model_name = self.model_engine.split("/", 1)
            required_params = ["model", "messages"]
            supported_params = get_supported_openai_params(
                model=_model_name, custom_llm_provider=_model_provider
            )
            params = required_params + (supported_params or [])
            completion_args = {
                k: v for k, v in completion_args.items() if k in params
            }

        # Handle reasoning models vs non-reasoning defaults
        if self._is_reasoning_model():
            if "temperature" in completion_args:
                logger.warning(
                    "Temperature parameter is not supported for reasoning models. Removing."
                )
                completion_args.pop("temperature", None)
            if reasoning_effort is not None:
                if reasoning_effort in ("low", "medium", "high"):
                    completion_args["reasoning_effort"] = reasoning_effort
                else:
                    logger.warning(
                        f"Invalid reasoning_effort '{reasoning_effort}'. Must be 'low', 'medium', or 'high'."
                    )
        else:
            completion_args.setdefault("temperature", 0.0)

        # Probe and cache support for temperature and reasoning_effort
        capabilities = self._get_capabilities()

        # Temperature
        if "temperature" in completion_args:
            temp_supported = capabilities.get("temperature")
            if temp_supported is False:
                completion_args.pop("temperature", None)
            elif temp_supported is None:
                try:
                    comp = completion(**completion_args)
                    self._set_capabilities({"temperature": True})
                    return _postprocess_content(comp.choices[0].message.content)
                except Exception as exc:
                    if self._is_unsupported_parameter_error(exc, "temperature"):
                        completion_args.pop("temperature", None)
                        self._set_capabilities({"temperature": False})
                    else:
                        raise

        # Reasoning effort
        if "reasoning_effort" in completion_args:
            re_supported = capabilities.get("reasoning_effort")
            if re_supported is False:
                completion_args.pop("reasoning_effort", None)
            elif re_supported is None:
                try:
                    comp = completion(**completion_args)
                    self._set_capabilities({"reasoning_effort": True})
                    return _postprocess_content(comp.choices[0].message.content)
                except Exception as exc:
                    if self._is_unsupported_parameter_error(
                        exc, "reasoning_effort"
                    ):
                        completion_args.pop("reasoning_effort", None)
                        self._set_capabilities({"reasoning_effort": False})
                    else:
                        raise

        # Final attempt with whatever parameters remain
        try:
            comp = completion(**completion_args)
            return _postprocess_content(comp.choices[0].message.content)
        except Exception as exc:
            # Last-resort targeted retry if unsupported parameter still slipped through
            removed_any = False
            for param in ("temperature", "reasoning_effort"):
                if (
                    param in completion_args
                    and self._is_unsupported_parameter_error(exc, param)
                ):
                    completion_args.pop(param, None)
                    removed_any = True
                    self._set_capabilities({param: False})
            if removed_any:
                comp = completion(**completion_args)
                return _postprocess_content(comp.choices[0].message.content)
            raise

    def _sanitize_reasoning_output(self, text: str) -> str:
        """
        Sanitize outputs from reasoning models (e.g., deepseek-r1) that prepend
        visible thinking traces like <think>...</think>. Also tries to
        extract JSON from code fences or trailing objects if present.
        """
        try:
            content = text if isinstance(text, str) else str(text)
            # Remove DeepSeek-style thinking blocks
            content = re.sub(
                r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE
            )

            trimmed = content.strip()

            # If content contains fenced blocks, prefer the last non-empty block
            fences = re.findall(
                r"```(?:json|python)?\s*([\s\S]*?)```",
                trimmed,
                flags=re.IGNORECASE,
            )
            for block in reversed(fences or []):
                candidate = block.strip()
                if candidate:
                    # If it parses as JSON, return the JSON block
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # Not strict JSON; still may be the intended payload
                        return candidate

            # Try to return a valid trailing JSON object or array if present
            for open_c, close_c in (("{", "}"), ("[", "]")):
                start = trimmed.rfind(open_c)
                end = trimmed.rfind(close_c)
                if start != -1 and end != -1 and end > start:
                    candidate2 = trimmed[start : end + 1].strip()
                    try:
                        json.loads(candidate2)
                        return candidate2
                    except Exception:
                        pass

            return trimmed
        except Exception:
            return text
