import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.exporter import (
    TruLensDBSpanExporter,
)

TRULENS_SERVICE_NAME = "trulens"


logger = logging.getLogger(__name__)


def init(session: TruSession, debug: bool = False):
    """Initialize the OpenTelemetry SDK with TruLens configuration."""
    resource = Resource.create({"service.name": TRULENS_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    session.experimental_enable_feature("otel_tracing")

    if session.connector:
        logger.debug("Exporting traces to the TruLens database")

        # Check the database revision
        try:
            db_revision = session.connector.db.get_db_revision()
            if db_revision is None:
                raise ValueError(
                    "Database revision is not set. Please run the migrations."
                )
            if int(db_revision) < 10:
                raise ValueError(
                    "Database revision is too low. Please run the migrations."
                )
        except Exception:
            raise ValueError("Error checking the database revision.")

        # Add the TruLens database exporter
        db_exporter = TruLensDBSpanExporter(session.connector)

        # When testing, use a simple span processor to avoid issues with batching/
        # asynchronous processing of the spans that results in the database not
        # being updated in time for the tests.
        db_processor = (
            SimpleSpanProcessor(db_exporter)
            if debug
            else BatchSpanProcessor(db_exporter)
        )
        provider.add_span_processor(db_processor)
