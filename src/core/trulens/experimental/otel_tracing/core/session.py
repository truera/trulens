import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import export as otel_export_sdk
from opentelemetry.trace.span import Span
from trulens.core import session as core_session
from trulens.core.database.connector import DBConnector
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core.exporter.connector import (
    TruLensOtelSpanExporter,
)
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.otel.semconv.trace import SpanAttributes

TRULENS_SERVICE_NAME = "trulens"

logger = logging.getLogger(__name__)


def _set_up_tracer_provider() -> TracerProvider:
    resource = Resource.create({"service.name": TRULENS_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    global_tracer_provider = trace.get_tracer_provider()
    if not isinstance(global_tracer_provider, TracerProvider):
        raise ValueError("Received a TracerProvider of an unexpected type!")
    return global_tracer_provider


def _can_import(to_import: str) -> bool:
    try:
        __import__(to_import)
        return True
    except ImportError:
        return False


class TrulensOtelSpanProcessor(otel_export_sdk.BatchSpanProcessor):
    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        set_general_span_attributes(
            span,
            span_type=SpanAttributes.SpanType.UNKNOWN,
            context=parent_context,
        )


class _TruSession(core_session.TruSession):
    def _validate_otel_exporter(
        self,
        exporter: Optional[otel_export_sdk.SpanExporter],
        connector: DBConnector,
    ) -> otel_export_sdk.SpanExporter:
        if (
            exporter is None
            and _can_import("trulens.connectors.snowflake")
            and _can_import("trulens.connectors.snowflake.otel_exporter")
        ):
            from trulens.connectors.snowflake import SnowflakeConnector
            from trulens.connectors.snowflake.otel_exporter import (
                TruLensSnowflakeSpanExporter,
            )

            if (
                isinstance(connector, SnowflakeConnector)
                and connector.use_account_event_table
            ):
                exporter = TruLensSnowflakeSpanExporter(connector)
        if not exporter:
            exporter = TruLensOtelSpanExporter(connector)
        if not isinstance(exporter, otel_export_sdk.SpanExporter):
            raise ValueError(
                "Provided exporter must be an OpenTelemetry SpanExporter!"
            )
        self._experimental_otel_exporter = exporter
        return exporter

    def _set_up_otel_exporter(
        self,
        connector: DBConnector,
        exporter: Optional[otel_export_sdk.SpanExporter],
    ):
        logger.info(
            f"{text_utils.UNICODE_CHECK} OpenTelemetry exporter set: "
            f"{python_utils.class_name(exporter.__class__)}"
        )

        tracer_provider = _set_up_tracer_provider()
        # Setting it here for easy access without having to assert the type
        # every time
        self._experimental_tracer_provider = tracer_provider
        exporter = _TruSession._validate_otel_exporter(
            self, exporter, connector
        )
        self._experimental_otel_span_processor = TrulensOtelSpanProcessor(
            exporter
        )
        tracer_provider.add_span_processor(
            self._experimental_otel_span_processor
        )
        logger.info(
            f"{text_utils.UNICODE_CHECK} Added new TrulensOtelSpanProcessor"
        )

    @staticmethod
    def _track_costs_for_module_member(
        module,
        method: str,
        cost_computer: Callable[[Any], Dict[str, Any]],
    ):
        from trulens.core.otel.instrument import instrument_cost_computer

        for cls in dir(module):
            obj = python_utils.safer_getattr(module, cls)
            if (
                obj is not None
                and isinstance(obj, type)
                and hasattr(obj, method)
            ):
                logger.info(
                    f"Instrumenting {obj.__name__}.{method} for cost tracking"
                )

                # Create an async-aware cost computer wrapper
                def cost_attributes(ret, exception, *args, **kwargs):
                    """Compute costs, handling both sync and async responses."""
                    logger.debug(
                        f"Cost computer called with return type: {type(ret)}"
                    )
                    try:
                        # The handle_response method now handles async responses internally
                        return cost_computer(ret)
                    except Exception as e:
                        # Only log as debug since costs might be computed elsewhere
                        logger.debug(f"Cost computation skipped: {e}")
                        return {}

                instrument_cost_computer(
                    obj,
                    method,
                    attributes=cost_attributes,
                )

    @staticmethod
    def _track_costs():
        if _can_import("trulens.providers.cortex.endpoint"):
            from snowflake.cortex._sse_client import SSEClient
            from trulens.core.otel.instrument import instrument_cost_computer
            from trulens.providers.cortex.endpoint import CortexCostComputer

            instrument_cost_computer(
                SSEClient,
                "events",
                attributes=lambda ret,
                exception,
                *args,
                **kwargs: CortexCostComputer.handle_response(ret),
            )
        if _can_import("trulens.providers.openai.endpoint"):
            import openai
            from openai import resources
            from openai.resources import chat
            from trulens.providers.openai.endpoint import OpenAICostComputer

            # The existing instrumentation handles sync calls
            for module in [openai, resources, chat]:
                _TruSession._track_costs_for_module_member(
                    module,
                    "create",
                    OpenAICostComputer.handle_response,
                )

            # Instrument AsyncOpenAI.post method directly for async cost tracking
            try:
                from openai import AsyncOpenAI
                from trulens.core.otel.instrument import instrument_method

                def async_post_cost_attributes(ret, exception, *args, **kwargs):
                    """Extract costs and capture input/output from AsyncOpenAI post responses."""

                    attrs = {}

                    # Capture the input (path and request body)
                    if args and len(args) > 0:
                        # First arg is usually the path
                        path = str(args[0]) if args[0] else "unknown"
                        attrs["openai.api.path"] = path

                    # Capture request body if present
                    if "body" in kwargs:
                        import json

                        try:
                            # Serialize the request body
                            if hasattr(kwargs["body"], "__dict__"):
                                body_dict = kwargs["body"].__dict__
                            else:
                                body_dict = kwargs["body"]

                            # Map request to standard attributes
                            if isinstance(body_dict, dict):
                                # Store messages - use custom attributes for LLM-specific fields
                                if "messages" in body_dict:
                                    attrs["llm.prompts"] = json.dumps(
                                        body_dict["messages"]
                                    )
                                    # Extract just the user message for a simpler view
                                    for msg in body_dict["messages"]:
                                        if (
                                            isinstance(msg, dict)
                                            and msg.get("role") == "user"
                                        ):
                                            attrs["llm.input_text"] = msg.get(
                                                "content", ""
                                            )
                                            break

                                # Store model if specified in request
                                if "model" in body_dict:
                                    attrs[SpanAttributes.COST.MODEL] = (
                                        body_dict["model"]
                                    )

                                # Store temperature and other parameters
                                if "temperature" in body_dict:
                                    attrs["llm.temperature"] = body_dict[
                                        "temperature"
                                    ]
                                if "max_tokens" in body_dict:
                                    attrs["llm.max_tokens"] = body_dict[
                                        "max_tokens"
                                    ]

                                # Store full request for debugging
                                attrs["openai.api.request"] = json.dumps(
                                    body_dict
                                )[:2000]  # Limit size
                        except Exception:
                            pass  # Silently skip serialization errors

                    # Capture the output
                    if ret:
                        try:
                            # Try to serialize the response
                            if hasattr(ret, "model_dump"):
                                # Pydantic model
                                output = ret.model_dump()
                            elif hasattr(ret, "__dict__"):
                                output = {
                                    k: v
                                    for k, v in ret.__dict__.items()
                                    if not k.startswith("_")
                                }
                            else:
                                output = str(ret)

                            import json

                            if isinstance(output, dict):
                                # Map to standard TruLens span attributes

                                # Model information
                                if output.get("model"):
                                    attrs[SpanAttributes.COST.MODEL] = output[
                                        "model"
                                    ]

                                # Token usage - these match the COST attributes
                                usage = output.get("usage", {})
                                if usage:
                                    attrs[
                                        SpanAttributes.COST.NUM_PROMPT_TOKENS
                                    ] = usage.get("prompt_tokens", 0)
                                    attrs[
                                        SpanAttributes.COST.NUM_COMPLETION_TOKENS
                                    ] = usage.get("completion_tokens", 0)
                                    attrs[SpanAttributes.COST.NUM_TOKENS] = (
                                        usage.get("total_tokens", 0)
                                    )

                                    # Also check for reasoning tokens (for o1 models)
                                    if "completion_tokens_details" in usage:
                                        details = usage[
                                            "completion_tokens_details"
                                        ]
                                        if "reasoning_tokens" in details:
                                            attrs[
                                                SpanAttributes.COST.NUM_REASONING_TOKENS
                                            ] = details["reasoning_tokens"]

                                # Response content
                                if output.get("choices"):
                                    first_choice = output["choices"][0]
                                    if isinstance(first_choice, dict):
                                        message = first_choice.get(
                                            "message", {}
                                        )
                                        if isinstance(message, dict):
                                            content = message.get("content", "")
                                            attrs[
                                                SpanAttributes.CALL.RETURN
                                            ] = content
                                            # Also store as LLM completion with custom attribute
                                            attrs["llm.completions"] = (
                                                json.dumps([
                                                    {
                                                        "role": message.get(
                                                            "role", "assistant"
                                                        ),
                                                        "content": content,
                                                    }
                                                ])
                                            )
                                            attrs["llm.output_text"] = content

                                # Store full response for debugging
                                summary = {
                                    "model": output.get("model", "unknown"),
                                    "usage": usage,
                                    "choices": len(output.get("choices", [])),
                                }
                                attrs["openai.api.response"] = json.dumps(
                                    summary
                                )
                            else:
                                attrs[SpanAttributes.CALL.RETURN] = str(output)[
                                    :1000
                                ]
                        except Exception:
                            pass  # Silently skip serialization errors
                            attrs[SpanAttributes.CALL.RETURN] = (
                                f"<{type(ret).__name__}>"
                            )

                    # Check if this is a chat completion response for cost tracking
                    # Only compute costs for actual ChatCompletion objects
                    if (
                        hasattr(ret, "model")
                        and hasattr(ret, "usage")
                        and ret.__class__.__name__
                        in ["ChatCompletion", "ParsedChatCompletion"]
                    ):
                        try:
                            cost_attrs = OpenAICostComputer.handle_response(ret)
                            attrs.update(cost_attrs)
                        except Exception as e:
                            # This should rarely happen now that we filter by type
                            logger.debug(
                                f"Unexpected cost computation error: {e}"
                            )

                    return attrs

                # Instrument the post method which is async and returns the actual response
                instrument_method(
                    AsyncOpenAI,
                    "post",
                    span_type=SpanAttributes.SpanType.GENERATION,
                    attributes=async_post_cost_attributes,
                )

            except ImportError as e:
                logger.debug(f"Could not instrument AsyncOpenAI: {e}")

        if _can_import("trulens.providers.litellm.endpoint"):
            import litellm
            from trulens.core.otel.instrument import instrument_method
            from trulens.providers.litellm.endpoint import LiteLLMCostComputer

            instrument_method(
                litellm,
                "completion",
                span_type=SpanAttributes.SpanType.GENERATION,
                attributes=lambda ret,
                exception,
                *args,
                **kwargs: LiteLLMCostComputer.handle_response(ret),
                must_be_first_wrapper=True,
            )
