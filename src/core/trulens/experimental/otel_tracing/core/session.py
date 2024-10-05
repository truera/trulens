from typing import Any, Optional

from trulens.core import experimental as mod_experimental
from trulens.core.session import TruSession
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils


class _TruSession(TruSession):
    def _setup_otel_exporter(
        self, val: Optional[Any]
    ):  # any actually otel_export_sdk.SpanExporter
        if val is None:
            self._experimental_otel_exporter = None
            return

        # from opentelemetry import sdk as otel_sdk
        try:
            from opentelemetry.sdk.trace import export as otel_export_sdk
        except ImportError:
            raise ImportError(
                "OpenTelemetry SDK not found. Please install OpenTelemetry SDK to use OpenTelemetry tracing."
            )

        assert isinstance(
            val, otel_export_sdk.SpanExporter
        ), "otel_exporter must be an OpenTelemetry SpanExporter."

        self._experimental_feature(
            flag=mod_experimental.Feature.OTEL_TRACING, value=True, lock=True
        )

        print(
            f"{text_utils.UNICODE_CHECK} OpenTelemetry exporter set: {python_utils.class_name(val.__class__)}"
        )

        self._experimental_otel_exporter = val
