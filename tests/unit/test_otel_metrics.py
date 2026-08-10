from importlib.util import find_spec
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.core.experimental import Feature
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.metrics import (
    OPERATION_DURATION_BUCKET_BOUNDARIES,
)
from trulens.experimental.otel_tracing.core.metrics import (
    OPERATION_DURATION_METRIC,
)
from trulens.experimental.otel_tracing.core.metrics import TOKEN_TYPE_ATTRIBUTE
from trulens.experimental.otel_tracing.core.metrics import (
    TOKEN_USAGE_BUCKET_BOUNDARIES,
)
from trulens.experimental.otel_tracing.core.metrics import TOKEN_USAGE_METRIC
from trulens.experimental.otel_tracing.core.metrics import (
    TrulensOtelMetricsSpanProcessor,
)
from trulens.experimental.otel_tracing.core.session import (
    _create_otlp_exporters,
)
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes


def _has_otlp_exporter() -> bool:
    """Return whether the optional OTLP exporter package is installed."""
    try:
        return find_spec("opentelemetry.exporter.otlp.proto.grpc") is not None
    except ModuleNotFoundError:
        # ``find_spec`` raises when the optional ``opentelemetry.exporter``
        # parent package is not installed at all.
        return False


class _Histogram:
    def __init__(self):
        self.records = []

    def record(self, value, attributes=None):
        self.records.append((value, dict(attributes or {})))


class _Meter:
    def __init__(self):
        self.histograms = {}

    def create_histogram(self, name, **kwargs):
        histogram = _Histogram()
        self.histograms[name] = histogram
        return histogram


class _SpanExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class _MetricExporter(MetricExporter):
    def __init__(self):
        super().__init__()
        self.metrics_data = []

    def export(self, metrics_data, timeout_millis=10000, **kwargs):
        self.metrics_data.append(metrics_data)
        return MetricExportResult.SUCCESS

    def shutdown(self, timeout_millis=30000, **kwargs):
        pass

    def force_flush(self, timeout_millis=10000):
        return True


class TestOtelMetricsSpanProcessor(unittest.TestCase):
    def test_force_flush_flushes_metrics_after_trace_timeout(self):
        class _Provider:
            def __init__(self, result):
                self.result = result
                self.calls = []

            def force_flush(self, timeout_millis):
                self.calls.append(timeout_millis)
                return self.result

        trace_provider = _Provider(False)
        meter_provider = _Provider(True)
        session = SimpleNamespace(
            experimental_feature=lambda feature: (
                feature == Feature.OTEL_TRACING
            ),
            _experimental_tracer_provider=trace_provider,
            _experimental_meter_provider=meter_provider,
        )

        self.assertFalse(TruSession.force_flush(session, 123))
        self.assertEqual(trace_provider.calls, [123])
        self.assertGreaterEqual(meter_provider.calls[0], 0)
        self.assertLessEqual(meter_provider.calls[0], 123)

    @mock.patch(
        "trulens.experimental.otel_tracing.core.session._create_otlp_exporters"
    )
    def test_otlp_public_option_wires_metrics_pipeline(self, create_exporters):
        TruSession.delete_singleton(TruSession)
        span_exporter = _SpanExporter()
        metric_exporter = _MetricExporter()
        create_exporters.return_value = (span_exporter, metric_exporter)
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as db:
            session = TruSession(
                database_url=f"sqlite:///{db.name}",
                otel_exporter="otlp",
                otlp_endpoint="http://localhost:4317",
            )
            create_exporters.assert_called_once_with("http://localhost:4317")
            tracer = session._experimental_tracer_provider.get_tracer(
                "test-otel-metrics"
            )
            with tracer.start_as_current_span("generation") as span:
                span.set_attribute(
                    SpanAttributes.SPAN_TYPE,
                    SpanAttributes.SpanType.GENERATION.value,
                )
                span.set_attribute(GenAIAttributes.OPERATION.NAME, "chat")
                span.set_attribute(GenAIAttributes.PROVIDER.NAME, "openai")
                span.set_attribute(GenAIAttributes.USAGE.INPUT_TOKENS, 2)
                span.set_attribute(GenAIAttributes.USAGE.OUTPUT_TOKENS, 1)
                span.set_attribute(GenAIAttributes.REQUEST.MODEL, "gpt-4o")
            self.assertTrue(session.force_flush())
            self.assertTrue(metric_exporter.metrics_data)
            self.assertEqual(
                metric_exporter.metrics_data[-1].resource_metrics[0].resource,
                session._experimental_tracer_provider.resource,
            )
            metric_names = {
                metric.name
                for resource_metrics in metric_exporter.metrics_data[
                    -1
                ].resource_metrics
                for scope_metrics in resource_metrics.scope_metrics
                for metric in scope_metrics.metrics
            }
            self.assertEqual(
                metric_names,
                {TOKEN_USAGE_METRIC, OPERATION_DURATION_METRIC},
            )
            session._experimental_meter_provider.shutdown()
            session.experimental_otel_exporter.shutdown()
        TruSession.delete_singleton(TruSession)

    def test_otlp_option_validation(self):
        TruSession.delete_singleton(TruSession)
        with self.assertRaises(ValueError):
            TruSession(otel_exporter="invalid")
        with self.assertRaises(ValueError):
            TruSession(otlp_endpoint="http://localhost:4317")
        TruSession.delete_singleton(TruSession)

    @unittest.skipUnless(
        _has_otlp_exporter(),
        "OTLP gRPC exporter is not installed",
    )
    def test_otlp_factory_creates_trace_and_metric_exporters(self):
        span_exporter, metric_exporter = _create_otlp_exporters(
            "http://localhost:4317"
        )

        self.assertEqual(type(span_exporter).__name__, "OTLPSpanExporter")
        self.assertEqual(type(metric_exporter).__name__, "OTLPMetricExporter")
        span_exporter.shutdown()
        metric_exporter.shutdown()

    def test_generation_span_emits_token_usage_and_duration(self):
        meter = _Meter()
        processor = TrulensOtelMetricsSpanProcessor(meter)
        span = SimpleNamespace(
            attributes={
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.GENERATION,
                GenAIAttributes.OPERATION.NAME: "chat",
                GenAIAttributes.SYSTEM.NAME: "openai",
                GenAIAttributes.REQUEST.MODEL: "gpt-4o",
                GenAIAttributes.USAGE.INPUT_TOKENS: 10,
                GenAIAttributes.USAGE.OUTPUT_TOKENS: 4,
            },
            start_time=1_000_000_000,
            end_time=3_500_000_000,
        )

        processor.on_end(span)

        token_records = meter.histograms[TOKEN_USAGE_METRIC].records
        self.assertEqual(len(token_records), 2)
        self.assertEqual(token_records[0][0], 10)
        self.assertEqual(token_records[0][1][TOKEN_TYPE_ATTRIBUTE], "input")
        self.assertEqual(token_records[1][0], 4)
        self.assertEqual(token_records[1][1][TOKEN_TYPE_ATTRIBUTE], "output")
        self.assertEqual(
            token_records[0][1][GenAIAttributes.OPERATION.NAME], "chat"
        )
        self.assertEqual(
            token_records[0][1][GenAIAttributes.PROVIDER.NAME], "openai"
        )

        duration_records = meter.histograms[OPERATION_DURATION_METRIC].records
        self.assertEqual(duration_records[0][0], 2.5)
        self.assertEqual(
            duration_records[0][1][GenAIAttributes.REQUEST.MODEL], "gpt-4o"
        )

    def test_trulens_cost_attributes_are_metric_fallbacks(self):
        meter = _Meter()
        processor = TrulensOtelMetricsSpanProcessor(meter)
        span = SimpleNamespace(
            attributes={
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.GENERATION,
                SpanAttributes.COST.MODEL: "provider-model",
                SpanAttributes.COST.NUM_PROMPT_TOKENS: 7,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS: 3,
            },
            start_time=0,
            end_time=1_000_000_000,
        )

        processor.on_end(span)

        input_record = meter.histograms[TOKEN_USAGE_METRIC].records[0]
        self.assertEqual(input_record[0], 7)
        self.assertEqual(
            input_record[1][GenAIAttributes.OPERATION.NAME], "generation"
        )
        self.assertEqual(
            input_record[1][GenAIAttributes.PROVIDER.NAME], "_OTHER"
        )
        self.assertEqual(
            input_record[1][GenAIAttributes.REQUEST.MODEL], "provider-model"
        )

    def test_non_generation_span_does_not_emit_metrics(self):
        meter = _Meter()
        processor = TrulensOtelMetricsSpanProcessor(meter)
        span = SimpleNamespace(
            attributes={
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.UNKNOWN,
            },
            start_time=0,
            end_time=1_000_000_000,
        )

        processor.on_end(span)

        self.assertEqual(meter.histograms[TOKEN_USAGE_METRIC].records, [])
        self.assertEqual(
            meter.histograms[OPERATION_DURATION_METRIC].records, []
        )

    def test_sdk_metrics_pipeline_emits_otel_histograms(self):
        reader = InMemoryMetricReader()
        meter_provider = MeterProvider(
            metric_readers=[reader],
            views=[
                # Exercise the same explicit boundaries used by the OTLP
                # session pipeline without constructing an OTLP exporter.
                View(
                    instrument_name=TOKEN_USAGE_METRIC,
                    aggregation=ExplicitBucketHistogramAggregation(
                        TOKEN_USAGE_BUCKET_BOUNDARIES
                    ),
                ),
                View(
                    instrument_name=OPERATION_DURATION_METRIC,
                    aggregation=ExplicitBucketHistogramAggregation(
                        OPERATION_DURATION_BUCKET_BOUNDARIES
                    ),
                ),
            ],
        )
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            TrulensOtelMetricsSpanProcessor(
                meter_provider.get_meter("test-otel-metrics")
            )
        )

        try:
            tracer = tracer_provider.get_tracer("test-otel-metrics")
            span = tracer.start_span("generation", start_time=1_000_000_000)
            span.set_attribute(
                SpanAttributes.SPAN_TYPE,
                SpanAttributes.SpanType.GENERATION.value,
            )
            span.set_attribute(GenAIAttributes.OPERATION.NAME, "chat")
            span.set_attribute(GenAIAttributes.PROVIDER.NAME, "openai")
            span.set_attribute(GenAIAttributes.USAGE.INPUT_TOKENS, 10)
            span.set_attribute(GenAIAttributes.USAGE.OUTPUT_TOKENS, 4)
            span.end(end_time=3_500_000_000)

            metrics_data = reader.get_metrics_data()
            metrics = {
                metric.name: metric
                for resource_metrics in metrics_data.resource_metrics
                for scope_metrics in resource_metrics.scope_metrics
                for metric in scope_metrics.metrics
            }
            self.assertEqual(
                set(metrics),
                {TOKEN_USAGE_METRIC, OPERATION_DURATION_METRIC},
            )
            self.assertEqual(
                sorted(
                    data_point.sum
                    for data_point in metrics[
                        TOKEN_USAGE_METRIC
                    ].data.data_points
                ),
                [4, 10],
            )
            self.assertEqual(
                metrics[OPERATION_DURATION_METRIC].data.data_points[0].sum,
                2.5,
            )
            self.assertEqual(
                metrics[TOKEN_USAGE_METRIC].data.data_points[0].explicit_bounds,
                TOKEN_USAGE_BUCKET_BOUNDARIES,
            )
            self.assertEqual(
                metrics[OPERATION_DURATION_METRIC]
                .data.data_points[0]
                .explicit_bounds,
                OPERATION_DURATION_BUCKET_BOUNDARIES,
            )
        finally:
            tracer_provider.shutdown()
            meter_provider.shutdown()


if __name__ == "__main__":
    unittest.main()
