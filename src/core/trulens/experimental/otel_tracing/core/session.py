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
from trulens.experimental.otel_tracing.core.exporter import (
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
        self._experimental_tracer_provider = provider

        # Export to the connector provided.
        provider.add_span_processor(
            otel_export_sdk.BatchSpanProcessor(
                TruLensOTELSpanExporter(connector)
            )
        )

        if exporter:
            assert isinstance(
                exporter, otel_export_sdk.SpanExporter
            ), "otel_exporter must be an OpenTelemetry SpanExporter."

            # When testing, use a simple span processor to avoid issues with batching/
            # asynchronous processing of the spans that results in the database not
            # being updated in time for the tests.
            db_processor = otel_export_sdk.BatchSpanProcessor(exporter)
            provider.add_span_processor(db_processor)
