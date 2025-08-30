import logging
from typing import Dict, Optional, Sequence, Type, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from trulens.feedback import llm_provider
from trulens.providers.langchain import endpoint as langchain_endpoint

logger = logging.getLogger(__name__)


def _convert_message(message: Union[Dict, BaseMessage]) -> BaseMessage:
    """Convert a message to a LangChain BaseMessage."""
    if isinstance(message, BaseMessage):
        return message
    if "role" not in message or message["role"] == "user":
        return HumanMessage(content=message["content"])
    return AIMessage(content=message["content"])


class Langchain(llm_provider.LLMProvider):
    """Out of the box feedback functions using LangChain LLMs and ChatModels

    Create a LangChain Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.langchain import LangChain
        from langchain_community.llms import OpenAI

        gpt3_llm = OpenAI(model="gpt-3.5-turbo-instruct")
        langchain_provider = LangChain(chain = gpt3_llm)
        ```

    Args:
        chain: LangChain LLM.
    """

    endpoint: langchain_endpoint.LangchainEndpoint

    def __init__(
        self,
        chain: Union[BaseLLM, BaseChatModel],
        *args,
        model_engine: str = "",
        **kwargs,
    ):
        self_kwargs = dict(kwargs)
        self_kwargs["model_engine"] = model_engine or type(chain).__name__
        self_kwargs["endpoint"] = langchain_endpoint.LangchainEndpoint(
            *args, chain=chain, **kwargs
        )

        super().__init__(**self_kwargs)

    # Capability probing and caching helpers come from LLMProvider
    def _capabilities_key(self) -> str:  # type: ignore[override]
        # Prefer explicit model_engine; fallback to LC class name
        return self.model_engine or type(self.endpoint.chain).__name__

    def _is_unsupported_parameter_error(
        self, exc: Exception, parameter: str
    ) -> bool:
        message = str(getattr(exc, "message", "")) or str(exc)
        lowered = message.lower()
        return (
            ("unexpected keyword" in lowered)
            or ("got an unexpected" in lowered)
            or ("does not support" in lowered)
            or ("is not allowed" in lowered)
            or ("unknown" in lowered)
        ) and (parameter in lowered)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Union[Dict, BaseMessage]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        # LangChain generally does not support structured outputs via response_format.
        # Ignore if provided to avoid unexpected keyword errors downstream.
        if response_format is not None:
            logger.debug(
                "Ignoring response_format in LangChain provider; not supported via chain.invoke."
            )

        call_kwargs = dict(kwargs)

        # Reasoning model handling mirrors OpenAI provider behavior.
        if self._is_reasoning_model():
            if "temperature" in call_kwargs:
                logger.warning(
                    "Temperature parameter is not supported for reasoning models in LangChain calls. Removing."
                )
                call_kwargs.pop("temperature", None)
            if reasoning_effort is not None:
                if reasoning_effort not in ("low", "medium", "high"):
                    logger.warning(
                        f"Invalid reasoning_effort '{reasoning_effort}'. Must be 'low', 'medium', or 'high'."
                    )
                else:
                    call_kwargs["reasoning_effort"] = reasoning_effort
        else:
            # Default to deterministic behavior for non-reasoning models
            call_kwargs.setdefault("temperature", 0.0)

        def _invoke_with_messages(_msgs_or_prompt):
            capabilities = self._get_capabilities()

            # Probe temperature support
            if "temperature" in call_kwargs:
                temperature_supported = capabilities.get("temperature")
                if temperature_supported is False:
                    call_kwargs.pop("temperature", None)
                elif temperature_supported is None:
                    try:
                        result = self.endpoint.chain.invoke(
                            _msgs_or_prompt, **call_kwargs
                        )
                        self._set_capabilities({"temperature": True})
                        return result
                    except TypeError as exc:
                        if self._is_unsupported_parameter_error(
                            exc, "temperature"
                        ):
                            call_kwargs.pop("temperature", None)
                            self._set_capabilities({"temperature": False})
                        else:
                            raise

            # Probe reasoning_effort support
            if "reasoning_effort" in call_kwargs:
                re_supported = capabilities.get("reasoning_effort")
                if re_supported is False:
                    call_kwargs.pop("reasoning_effort", None)
                elif re_supported is None:
                    try:
                        result = self.endpoint.chain.invoke(
                            _msgs_or_prompt, **call_kwargs
                        )
                        self._set_capabilities({"reasoning_effort": True})
                        return result
                    except TypeError as exc:
                        if self._is_unsupported_parameter_error(
                            exc, "reasoning_effort"
                        ):
                            call_kwargs.pop("reasoning_effort", None)
                            self._set_capabilities({"reasoning_effort": False})
                        else:
                            raise

            # Final attempt with whatever parameters remain
            try:
                return self.endpoint.chain.invoke(
                    _msgs_or_prompt, **call_kwargs
                )
            except Exception as exc:
                # Last-resort targeted retry if unsupported parameter still slipped through
                removed_any = False
                for param in ("temperature", "reasoning_effort"):
                    if (
                        param in call_kwargs
                        and self._is_unsupported_parameter_error(exc, param)
                    ):
                        call_kwargs.pop(param, None)
                        removed_any = True
                        # Cache negative support
                        self._set_capabilities({param: False})
                if removed_any:
                    return self.endpoint.chain.invoke(
                        _msgs_or_prompt, **call_kwargs
                    )
                raise

        if prompt is not None:
            predict = _invoke_with_messages(prompt)
        elif messages is not None:
            lc_messages: list[BaseMessage] = [
                _convert_message(message) for message in messages
            ]
            predict = _invoke_with_messages(lc_messages)
            if isinstance(self.endpoint.chain, BaseChatModel):
                if not isinstance(predict, BaseMessage):
                    raise ValueError(
                        "`chain.invoke` did not return a `langchain_core.messages.BaseMessage` as expected!"
                    )
                predict = predict.content
            elif isinstance(self.endpoint.chain, BaseLLM):
                if not isinstance(predict, str):
                    raise ValueError(
                        "`chain.invoke` did not return a `str` as expected!"
                    )
            else:
                raise ValueError("Unexpected `chain` type!")
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return predict
