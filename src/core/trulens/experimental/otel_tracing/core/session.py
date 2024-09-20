from trulens.core import experimental as mod_experimental
from trulens.core.session import TruSession
from trulens.core.utils import text as text_utils


class _TruSession(TruSession):
    def _setup_otel_exporter(self, _experimental_otel_exporter):
        self._experimental_assert_feature(mod_experimental.Feature.OTEL_TRACING)

        # from opentelemetry import sdk as otel_sdk
        try:
            from opentelemetry.sdk.trace import export as otel_export_sdk
        except ImportError:
            raise ImportError(
                "OpenTelemetry SDK not found. Please install OpenTelemetry SDK to use OpenTelemetry tracing."
            )

        assert isinstance(
            _experimental_otel_exporter, otel_export_sdk.SpanExporter
        ), "otel_exporter must be an OpenTelemetry SpanExporter."

        print(
            f"{text_utils.UNICODE_CHECK} OpenTelemetry exporter set: {_experimental_otel_exporter.__class__.__name__}"
        )

        self._experimental_otel_exporter = _experimental_otel_exporter
