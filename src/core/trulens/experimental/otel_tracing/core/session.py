import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import export as otel_export_sdk
from trulens.core import experimental as core_experimental
from trulens.core import session as core_session
from trulens.core.database.connector import DBConnector
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core.exporter.connector import (
    TruLensOTELSpanExporter,
)

TRULENS_SERVICE_NAME = "trulens"

logger = logging.getLogger(__name__)


class _TruSession(core_session.TruSession):
    def _setup_otel_exporter(
        self,
        connector: DBConnector,
        exporter: Optional[otel_export_sdk.SpanExporter],
    ):
        self._experimental_feature(
            flag=core_experimental.Feature.OTEL_TRACING, value=True, freeze=True
        )

        logger.info(
            f"{text_utils.UNICODE_CHECK} OpenTelemetry exporter set: {python_utils.class_name(exporter.__class__)}"
        )

        self._experimental_otel_exporter = exporter

        """Initialize the OpenTelemetry SDK with TruLens configuration."""
        resource = Resource.create({"service.name": TRULENS_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        global_trace_provider = trace.get_tracer_provider()
        if not isinstance(global_trace_provider, TracerProvider):
            raise ValueError("Received a TracerProvider of an unexpected type!")

        tracer_provider: TracerProvider = global_trace_provider

        # Setting it here for easy access without having to assert the type every time
        self._experimental_tracer_provider = tracer_provider

        if exporter and not isinstance(exporter, otel_export_sdk.SpanExporter):
            raise ValueError(
                "Provided exporter must be an OpenTelemetry SpanExporter"
            )

        db_processor = otel_export_sdk.BatchSpanProcessor(
            exporter if exporter else TruLensOTELSpanExporter(connector)
        )
        tracer_provider.add_span_processor(db_processor)
