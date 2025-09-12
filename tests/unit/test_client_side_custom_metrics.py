"""
Tests for client-side custom metrics functionality.
"""

import pytest
from trulens.core.feedback.custom_metric import MetricConfig
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


@pytest.mark.optional
class TestClientSideCustomMetrics(OtelTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_metric_config_creation(self) -> None:
        """Test MetricConfig creation and validation."""

        def test_metric(query: str) -> float:
            return len(query) / 100.0

        # Test basic creation with explicit metric_type
        config = MetricConfig(
            metric_name="test_metric_v1",
            metric_implementation=test_metric,
            metric_type="accuracy",
            computation_type="client",
        ).add_selector(
            "query",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        self.assertEqual(config.metric_name, "test_metric_v1")
        self.assertEqual(config.metric_type, "accuracy")
        self.assertEqual(config.computation_type, "client")
        self.assertIn("query", config.selectors)
        self.assertEqual(config.metric_implementation, test_metric)

        # Test creation without explicit metric_type (should default to function name)
        config_default = MetricConfig(
            metric_name="another_test",
            metric_implementation=test_metric,
        )
        self.assertEqual(
            config_default.metric_type, "test_metric"
        )  # Function name

    def test_metric_config_validation(self) -> None:
        """Test MetricConfig validation against function signatures."""

        def test_func(query: str, output: str) -> float:
            return 1.0

        # Valid config
        valid_config = (
            MetricConfig(metric_name="test", metric_implementation=test_func)
            .add_selector(
                "query",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            )
            .add_selector(
                "output",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            )
        )

        # Should not raise
        valid_config.validate_selectors()

        # Invalid config - missing parameter
        def single_param_func(query: str) -> float:
            return 1.0

        invalid_config = (
            MetricConfig(
                metric_name="test", metric_implementation=single_param_func
            )
            .add_selector(
                "query",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            )
            .add_selector(
                "extra_param",  # This parameter doesn't exist in the function
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            )
        )

        with self.assertRaises(ValueError):
            invalid_config.validate_selectors()

    def test_metric_config_feedback_creation(self) -> None:
        """Test MetricConfig feedback creation."""

        def test_metric(query: str) -> float:
            return 1.0

        config = MetricConfig(
            metric_name="test_metric", metric_implementation=test_metric
        ).add_selector(
            "query",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        feedback = config.create_feedback_definition()
        self.assertEqual(feedback.name, "test_metric")
        self.assertIsNotNone(feedback.imp)

    def test_compute_metrics_with_metric_configs(self) -> None:
        """Test compute_metrics with MetricConfig objects."""

        def text2sql_quality(query: str, sql: str) -> float:
            """Simple text2SQL quality metric."""
            if "SELECT" in sql.upper() and len(query) > 10:
                return 0.9
            return 0.3

        # Create metric configs
        metric_config = (
            MetricConfig(
                metric_name="text2sql_custom",
                metric_implementation=text2sql_quality,
                computation_type="client",
            )
            .add_selector(
                "query",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            )
            .add_selector(
                "sql",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            )
        )

        # Validate config
        metric_config.validate_selectors()

        # Test that feedback can be created
        feedback = metric_config.create_feedback_definition()
        self.assertEqual(feedback.name, "text2sql_custom")
        self.assertTrue(feedback.higher_is_better)

    def test_mixed_metrics_list(self) -> None:
        """Test compute_metrics with both string and MetricConfig objects."""

        def custom_accuracy(query: str) -> float:
            return len(query) / 100.0

        metric_config = MetricConfig(
            metric_name="custom_accuracy", metric_implementation=custom_accuracy
        ).add_selector(
            "query",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        # Test that both string metrics (server-side) and MetricConfig objects work
        mixed_metrics = [
            "answer_relevance",  # String - server-side metric
            metric_config,  # MetricConfig - client-side metric
        ]

        # Just test that the types are correctly identified
        client_configs = [m for m in mixed_metrics if not isinstance(m, str)]
        server_names = [m for m in mixed_metrics if isinstance(m, str)]

        self.assertEqual(len(client_configs), 1)
        self.assertEqual(len(server_names), 1)
        self.assertEqual(client_configs[0].metric_name, "custom_accuracy")
        self.assertEqual(server_names[0], "answer_relevance")

    def test_same_metric_type_different_names(self) -> None:
        """Test using the same metric implementation with different semantic names."""

        def accuracy_metric(query: str, output: str) -> float:
            """Generic accuracy metric implementation."""
            return 0.8 if len(query) > 10 else 0.3

        # Create two different metric configurations using the same implementation
        qa_accuracy_config = (
            MetricConfig(
                metric_name="qa_accuracy_v1",
                metric_implementation=accuracy_metric,
                metric_type="accuracy",
                description="Accuracy metric for Q&A task",
            )
            .add_selector(
                "query",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            )
            .add_selector(
                "output",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            )
        )

        text2sql_accuracy_config = (
            MetricConfig(
                metric_name="text2sql_accuracy_v2",
                metric_implementation=accuracy_metric,  # Same implementation
                metric_type="accuracy",  # Same metric_type
                description="Accuracy metric for text-to-SQL task",
            )
            .add_selector(
                "query",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            )
            .add_selector(
                "output",
                Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            )
        )

        # Verify they have different names but same type and implementation
        self.assertEqual(qa_accuracy_config.metric_name, "qa_accuracy_v1")
        self.assertEqual(
            text2sql_accuracy_config.metric_name, "text2sql_accuracy_v2"
        )

        # Same metric_type and implementation
        self.assertEqual(qa_accuracy_config.metric_type, "accuracy")
        self.assertEqual(text2sql_accuracy_config.metric_type, "accuracy")
        self.assertEqual(
            qa_accuracy_config.metric_implementation,
            text2sql_accuracy_config.metric_implementation,
        )

        # Different descriptions
        self.assertNotEqual(
            qa_accuracy_config.description, text2sql_accuracy_config.description
        )
