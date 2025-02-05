import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import export as otel_export_sdk
from trulens.core import session as core_session
from trulens.core.database.connector import DBConnector
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core.exporter.connector import (
    TruLensOTELSpanExporter,
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

            if isinstance(connector, SnowflakeConnector):
                exporter = TruLensSnowflakeSpanExporter(connector)
        if not exporter:
            exporter = TruLensOTELSpanExporter(connector)
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
            f"{text_utils.UNICODE_CHECK} OpenTelemetry exporter set: {python_utils.class_name(exporter.__class__)}"
        )

        tracer_provider = _set_up_tracer_provider()
        # Setting it here for easy access without having to assert the type every time
        self._experimental_tracer_provider = tracer_provider

        exporter = _TruSession._validate_otel_exporter(
            self, exporter, connector
        )

        self._experimental_otel_span_processor = (
            otel_export_sdk.BatchSpanProcessor(exporter)
        )
        tracer_provider.add_span_processor(
            self._experimental_otel_span_processor
        )

    @staticmethod
    def _track_costs_for_module_member(
        module,
        method: str,
        cost_computer: Callable[[Any], Dict[str, Any]],
        span_type: SpanAttributes.SpanType,
    ):
        from trulens.core.otel.instrument import instrument_method

        for cls in dir(module):
            obj = python_utils.safer_getattr(module, cls)
            if (
                obj is not None
                and isinstance(obj, type)
                and hasattr(obj, method)
            ):
                instrument_method(
                    obj,
                    method,
                    span_type=span_type,
                    full_scoped_attributes=lambda ret,
                    exception,
                    *args,
                    **kwargs: cost_computer(ret),
                    must_be_first_wrapper=True,
                )

    @staticmethod
    def _track_costs():
        if _can_import("trulens.providers.cortex.endpoint"):
            from snowflake.cortex._sse_client import SSEClient
            from trulens.core.otel.instrument import instrument_method
            from trulens.providers.cortex.endpoint import CortexCostComputer

            instrument_method(
                SSEClient,
                "events",
                span_type=SpanAttributes.SpanType.UNKNOWN,
                full_scoped_attributes=lambda ret,
                exception,
                *args,
                **kwargs: CortexCostComputer.handle_response(ret),
                must_be_first_wrapper=True,
            )
        if _can_import("trulens.providers.openai.endpoint"):
            import openai
            from openai import resources
            from openai.resources import chat
            from trulens.providers.openai.endpoint import OpenAICostComputer

            for module in [openai, resources, chat]:
                _TruSession._track_costs_for_module_member(
                    module,
                    "create",
                    OpenAICostComputer.handle_response,
                    SpanAttributes.SpanType.UNKNOWN,
                )
        if _can_import("trulens.providers.litellm.endpoint"):
            import litellm
            from trulens.core.otel.instrument import instrument_method
            from trulens.providers.litellm.endpoint import LiteLLMCostComputer

            instrument_method(
                litellm,
                "completion",
                span_type=SpanAttributes.SpanType.GENERATION,
                full_scoped_attributes=lambda ret,
                exception,
                *args,
                **kwargs: LiteLLMCostComputer.handle_response(ret),
                must_be_first_wrapper=True,
            )
