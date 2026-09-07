"""OpenTelemetry metric production for TruLens GenAI spans."""

from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor
from trulens.otel.semconv.trace import ErrorAttributes
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ServerAttributes
from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter


TOKEN_USAGE_METRIC = "gen_ai.client.token.usage"
OPERATION_DURATION_METRIC = "gen_ai.client.operation.duration"
TOKEN_TYPE_ATTRIBUTE = "gen_ai.token.type"

# Recommended explicit boundaries from the OpenTelemetry GenAI metric
# semantic conventions. These are applied through SDK Views so the feature
# remains compatible with the project's OTel API lower bound, which predates
# the create_histogram explicit_bucket_boundaries_advisory parameter.
TOKEN_USAGE_BUCKET_BOUNDARIES = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
OPERATION_DURATION_BUCKET_BOUNDARIES = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)


class TrulensOtelMetricsSpanProcessor(SpanProcessor):
    """Record GenAI metrics from completed TruLens generation spans."""

    def __init__(self, meter: "Meter") -> None:
        self._token_usage = meter.create_histogram(
            TOKEN_USAGE_METRIC,
            unit="{token}",
            description=(
                "Number of input and output tokens used by GenAI operations."
            ),
        )
        self._operation_duration = meter.create_histogram(
            OPERATION_DURATION_METRIC,
            unit="s",
            description="Duration of GenAI client operations.",
        )

    @staticmethod
    def _is_generation_span(span: ReadableSpan) -> bool:
        span_type = (span.attributes or {}).get(SpanAttributes.SPAN_TYPE)
        return span_type in (
            SpanAttributes.SpanType.GENERATION,
            SpanAttributes.SpanType.GENERATION.value,
        )

    @staticmethod
    def _metric_attributes(
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        metric_attributes: dict[str, Any] = {
            GenAIAttributes.OPERATION.NAME: attributes.get(
                GenAIAttributes.OPERATION.NAME, "generation"
            ),
            GenAIAttributes.PROVIDER.NAME: attributes.get(
                GenAIAttributes.PROVIDER.NAME,
                attributes.get(GenAIAttributes.SYSTEM.NAME, "_OTHER"),
            ),
        }

        model = attributes.get(GenAIAttributes.REQUEST.MODEL)
        if model is None:
            model = attributes.get(SpanAttributes.COST.MODEL)
        if model is not None:
            metric_attributes[GenAIAttributes.REQUEST.MODEL] = model

        for key in (
            GenAIAttributes.RESPONSE.MODEL,
            ServerAttributes.ADDRESS,
            ServerAttributes.PORT,
            ErrorAttributes.TYPE,
        ):
            if key in attributes:
                metric_attributes[key] = attributes[key]

        return metric_attributes

    @staticmethod
    def _token_count(
        attributes: dict[str, Any],
        gen_ai_attribute: str,
        trulens_attribute: str,
    ) -> Any:
        value = attributes.get(gen_ai_attribute)
        if value is None:
            value = attributes.get(trulens_attribute)
        return value

    def on_end(self, span: ReadableSpan) -> None:
        if not self._is_generation_span(span):
            return

        attributes = dict(span.attributes or {})
        metric_attributes = self._metric_attributes(attributes)
        for token_type, gen_ai_attribute, trulens_attribute in (
            (
                "input",
                GenAIAttributes.USAGE.INPUT_TOKENS,
                SpanAttributes.COST.NUM_PROMPT_TOKENS,
            ),
            (
                "output",
                GenAIAttributes.USAGE.OUTPUT_TOKENS,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS,
            ),
        ):
            token_count = self._token_count(
                attributes, gen_ai_attribute, trulens_attribute
            )
            if isinstance(token_count, int) and not isinstance(
                token_count, bool
            ):
                token_attributes = dict(metric_attributes)
                token_attributes[TOKEN_TYPE_ATTRIBUTE] = token_type
                self._token_usage.record(
                    token_count,
                    attributes=token_attributes,
                )

        if span.start_time is None or span.end_time is None:
            return

        duration_seconds = max(
            0.0, (span.end_time - span.start_time) / 1_000_000_000
        )
        self._operation_duration.record(
            duration_seconds,
            attributes=metric_attributes,
        )
