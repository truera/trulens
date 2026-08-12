"""Generic LLM Auto-Instrumentation for TruLens.

Auto-instruments SDK client completion methods for OpenAI, Anthropic,
Google GenAI, Bedrock, and LiteLLM to emit SpanType.GENERATION spans
with official gen_ai.* attributes, cost tracking, and span events.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

_LLM_AUTO_INSTRUMENTED: bool = False


def is_auto_instrumentation_enabled() -> bool:
    """Check if generic LLM auto-instrumentation is active."""
    return _LLM_AUTO_INSTRUMENTED


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def instrument_openai() -> None:
    if not _can_import("openai"):
        return
    try:
        from openai.resources import chat
        from trulens.core.otel.instrument import instrument_method
        from trulens.providers.openai.endpoint import OpenAICostComputer

        def openai_attributes(
            ret: Any, exception: Exception | None, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            attrs: dict[str, Any] = {
                "provider_name": "openai",
                "gen_ai.system": "openai",
            }

            model = kwargs.get("model")
            if not model and hasattr(ret, "model"):
                model = ret.model
            if model:
                attrs[SpanAttributes.COST.MODEL] = model
                attrs["gen_ai.request.model"] = model

            if ret is not None:
                try:
                    cost_attrs = OpenAICostComputer.handle_response(ret)
                    attrs.update(cost_attrs)
                except Exception as e:
                    logger.debug(f"OpenAI cost computation error: {e}")

                if hasattr(ret, "choices") and ret.choices:
                    first_choice = ret.choices[0]
                    if hasattr(first_choice, "message") and hasattr(
                        first_choice.message, "content"
                    ):
                        attrs["completion"] = first_choice.message.content

            if "messages" in kwargs:
                attrs["prompt"] = kwargs["messages"]
            elif "prompt" in kwargs:
                attrs["prompt"] = kwargs["prompt"]

            return attrs

        for cls in [chat.Completions, chat.AsyncCompletions]:
            try:
                instrument_method(
                    cls,
                    "create",
                    span_type=SpanAttributes.SpanType.GENERATION,
                    attributes=openai_attributes,
                )
            except Exception as e:
                logger.debug(f"Could not instrument {cls.__name__}.create: {e}")

    except Exception as e:
        logger.debug(f"OpenAI auto-instrumentation skipped: {e}")


def instrument_anthropic() -> None:
    if not _can_import("anthropic"):
        return
    try:
        from anthropic.resources import messages
        from trulens.core.otel.instrument import instrument_method

        def anthropic_attributes(
            ret: Any, exception: Exception | None, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            attrs: dict[str, Any] = {
                "provider_name": "anthropic",
                "gen_ai.system": "anthropic",
            }

            model = kwargs.get("model")
            if not model and hasattr(ret, "model"):
                model = ret.model
            if model:
                attrs[SpanAttributes.COST.MODEL] = model
                attrs["gen_ai.request.model"] = model

            if ret is not None and hasattr(ret, "usage"):
                usage = ret.usage
                if hasattr(usage, "input_tokens"):
                    attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] = (
                        usage.input_tokens
                    )
                if hasattr(usage, "output_tokens"):
                    attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = (
                        usage.output_tokens
                    )

            if "messages" in kwargs:
                attrs["prompt"] = kwargs["messages"]

            if ret is not None and hasattr(ret, "content") and ret.content:
                text_blocks = [
                    b.text for b in ret.content if hasattr(b, "text")
                ]
                if text_blocks:
                    attrs["completion"] = "\n".join(text_blocks)

            return attrs

        for cls in [messages.Messages, messages.AsyncMessages]:
            try:
                instrument_method(
                    cls,
                    "create",
                    span_type=SpanAttributes.SpanType.GENERATION,
                    attributes=anthropic_attributes,
                )
            except Exception as e:
                logger.debug(f"Could not instrument Anthropic {cls}: {e}")

    except Exception as e:
        logger.debug(f"Anthropic auto-instrumentation skipped: {e}")


def instrument_google() -> None:
    if not _can_import("google.genai"):
        return
    try:
        from google.genai import models
        from trulens.core.otel.instrument import instrument_method
        from trulens.providers.google.endpoint import GoogleCostComputer

        def google_attributes(
            ret: Any, exception: Exception | None, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            attrs: dict[str, Any] = {
                "provider_name": "google",
                "gen_ai.system": "google",
            }

            model = kwargs.get("model")
            if not model and args and isinstance(args[0], str):
                model = args[0]
            if model:
                attrs[SpanAttributes.COST.MODEL] = model
                attrs["gen_ai.request.model"] = model

            if ret is not None:
                try:
                    cost_attrs = GoogleCostComputer.handle_response(ret)
                    attrs.update(cost_attrs)
                except Exception as e:
                    logger.debug(f"Google GenAI cost computation error: {e}")

                if hasattr(ret, "text"):
                    attrs["completion"] = ret.text

            if "contents" in kwargs:
                attrs["prompt"] = kwargs["contents"]

            return attrs

        for cls in [models.Models, models.AsyncModels]:
            try:
                instrument_method(
                    cls,
                    "generate_content",
                    span_type=SpanAttributes.SpanType.GENERATION,
                    attributes=google_attributes,
                )
            except Exception as e:
                logger.debug(f"Could not instrument Google GenAI {cls}: {e}")

    except Exception as e:
        logger.debug(f"Google GenAI auto-instrumentation skipped: {e}")


def instrument_bedrock() -> None:
    if not _can_import("botocore"):
        return
    try:
        import botocore.client
        from trulens.core.otel.instrument import instrument_method

        def bedrock_attributes(
            ret: Any, exception: Exception | None, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            if not args or args[0] not in (
                "InvokeModel",
                "InvokeModelWithResponseStream",
            ):
                return {}

            params = args[1] if len(args) > 1 else kwargs.get("api_params", {})
            model_id = (
                params.get("modelId", "unknown")
                if isinstance(params, dict)
                else "unknown"
            )

            attrs: dict[str, Any] = {
                "provider_name": "aws.bedrock",
                "gen_ai.system": "aws.bedrock",
                SpanAttributes.COST.MODEL: model_id,
                "gen_ai.request.model": model_id,
            }

            if isinstance(params, dict):
                body = params.get("body")
                if body:
                    try:
                        if isinstance(body, (bytes, bytearray)):
                            body_dict = json.loads(body.decode("utf-8"))
                        elif isinstance(body, str):
                            body_dict = json.loads(body)
                        else:
                            body_dict = body
                        if isinstance(body_dict, dict):
                            if "prompt" in body_dict:
                                attrs["prompt"] = body_dict["prompt"]
                            elif "messages" in body_dict:
                                attrs["prompt"] = body_dict["messages"]
                    except Exception as e:
                        logger.debug(f"Error parsing Bedrock body: {e}")

            return attrs

        try:
            instrument_method(
                botocore.client.BaseClient,
                "_make_api_call",
                span_type=SpanAttributes.SpanType.GENERATION,
                attributes=bedrock_attributes,
            )
        except Exception as e:
            logger.debug(f"Could not instrument botocore BaseClient: {e}")

    except Exception as e:
        logger.debug(f"Bedrock auto-instrumentation skipped: {e}")


def instrument_litellm() -> None:
    if not _can_import("litellm"):
        return
    try:
        import litellm
        from trulens.core.otel.instrument import instrument_method
        from trulens.providers.litellm.endpoint import LiteLLMCostComputer

        def litellm_attributes(
            ret: Any, exception: Exception | None, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            attrs: dict[str, Any] = {
                "provider_name": "litellm",
                "gen_ai.system": "litellm",
            }

            model = kwargs.get("model")
            if not model and hasattr(ret, "model"):
                model = ret.model
            if model:
                attrs[SpanAttributes.COST.MODEL] = model
                attrs["gen_ai.request.model"] = model

            if ret is not None:
                try:
                    cost_attrs = LiteLLMCostComputer.handle_response(ret)
                    attrs.update(cost_attrs)
                except Exception as e:
                    logger.debug(f"LiteLLM cost computation error: {e}")

                if hasattr(ret, "choices") and ret.choices:
                    first_choice = ret.choices[0]
                    if hasattr(first_choice, "message") and hasattr(
                        first_choice.message, "content"
                    ):
                        attrs["completion"] = first_choice.message.content

            if "messages" in kwargs:
                attrs["prompt"] = kwargs["messages"]

            return attrs

        try:
            instrument_method(
                litellm,
                "completion",
                span_type=SpanAttributes.SpanType.GENERATION,
                attributes=litellm_attributes,
                must_be_first_wrapper=True,
            )
        except Exception as e:
            logger.debug(f"Could not instrument litellm.completion: {e}")

    except Exception as e:
        logger.debug(f"LiteLLM auto-instrumentation skipped: {e}")


def auto_instrument_all_llms() -> None:
    """Instrument all detected LLM SDK client classes for full GENERATION span emission."""
    global _LLM_AUTO_INSTRUMENTED
    if _LLM_AUTO_INSTRUMENTED:
        return

    instrument_openai()
    instrument_anthropic()
    instrument_google()
    instrument_bedrock()
    instrument_litellm()

    _LLM_AUTO_INSTRUMENTED = True
    logger.info("Generic LLM auto-instrumentation enabled.")
