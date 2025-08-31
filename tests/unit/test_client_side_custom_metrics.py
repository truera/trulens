"""
Tests for client-side custom metrics functionality.
"""

import pytest
from trulens.apps.app import TruApp
from trulens.core.feedback.custom_metric import EvaluationConfig
from trulens.core.feedback.custom_metric import _pending_metrics
from trulens.core.feedback.custom_metric import _registered_apps
from trulens.core.feedback.custom_metric import custom_metric
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


@pytest.mark.optional
class TestClientSideCustomMetrics(OtelTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Clear global registries for clean test state
        global _pending_metrics, _registered_apps
        _pending_metrics.clear()
        _registered_apps.clear()

    def test_evaluation_config_creation(self) -> None:
        """Test EvaluationConfig creation and validation."""
        # Test basic creation
        config = EvaluationConfig(
            metric_type="test_metric", computation_type="client"
        ).add_selector(
            "query",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        self.assertEqual(config.metric_type, "test_metric")
        self.assertEqual(config.computation_type, "client")
        self.assertIn("query", config.selectors)

        # Test from_dict creation
        config_dict = {
            "metric_type": "dict_metric",
            "computation_type": "client",
            "selectors": {
                "input": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                )
            },
        }
        config_from_dict = EvaluationConfig.from_dict(config_dict)
        self.assertEqual(config_from_dict.metric_type, "dict_metric")

    def test_evaluation_config_validation(self) -> None:
        """Test EvaluationConfig validation against function signatures."""

        def test_func(query: str, output: str) -> float:
            return 1.0

        # Valid config
        valid_config = (
            EvaluationConfig(metric_type="test")
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
        valid_config.validate_for_function(test_func)

        # Invalid config - missing parameter
        invalid_config = EvaluationConfig(metric_type="test").add_selector(
            "query",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        with self.assertRaises(ValueError):
            invalid_config.validate_for_function(test_func)

    def test_custom_metric_decorator(self) -> None:
        """Test @custom_metric decorator functionality."""

        @custom_metric(metric_type="test_accuracy", higher_is_better=True)
        def test_accuracy(query: str) -> float:
            return len(query) / 100.0

        self.assertEqual(test_accuracy.metric_type, "test_accuracy")
        self.assertTrue(test_accuracy.higher_is_better)
        self.assertEqual(test_accuracy(query="test"), 0.04)

        # Test without explicit metric_type
        @custom_metric()
        def another_metric(input: str) -> float:
            return 0.5

        self.assertEqual(another_metric.metric_type, "another_metric")

    def test_custom_metric_feedback_creation(self) -> None:
        """Test CustomMetric feedback creation."""

        @custom_metric(metric_type="test_metric")
        def test_metric(query: str) -> float:
            return 1.0

        selectors = {
            "query": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            )
        }

        feedback = test_metric.create_feedback_definition(selectors)
        self.assertEqual(feedback.name, "test_metric")
        self.assertIsNotNone(feedback.imp)

    def test_bidirectional_auto_registration(self) -> None:
        """Test bidirectional auto-registration works regardless of order."""

        # Scenario 1: Define metric before TruApp
        @custom_metric(metric_type="metric_before_app", auto_register=True)
        def metric_before_app(query: str) -> float:
            return 0.8

        # Check metric is in pending
        self.assertTrue(
            any(
                m[0].metric_type == "metric_before_app"
                for m in _pending_metrics
            )
        )

        # Create TruApp - should register pending metric
        class TestApp:
            @instrument()
            def query(self, query: str) -> str:
                return f"Response: {query}"

        app = TestApp()
        tru_app = TruApp(
            app, app_name="TestApp", app_version="v1", main_method=app.query
        )

        # Check metric was registered with app
        custom_metrics = tru_app.get_custom_metrics()
        metric_names = [m["metric"].metric_type for m in custom_metrics]
        self.assertIn("metric_before_app", metric_names)

        # Scenario 2: Define metric after TruApp
        @custom_metric(metric_type="metric_after_app", auto_register=True)
        def metric_after_app(query: str) -> float:
            return 0.9

        # Check metric was automatically registered with existing app
        updated_metrics = tru_app.get_custom_metrics()
        updated_metric_names = [
            m["metric"].metric_type for m in updated_metrics
        ]
        self.assertIn("metric_after_app", updated_metric_names)

    def test_auto_registration_disabled(self) -> None:
        """Test that auto_register=False prevents automatic registration."""

        @custom_metric(metric_type="manual_metric", auto_register=False)
        def manual_metric(query: str) -> float:
            return 0.7

        # Should not be in pending metrics
        self.assertFalse(
            any(m[0].metric_type == "manual_metric" for m in _pending_metrics)
        )

        # Create TruApp
        class TestApp:
            @instrument()
            def query(self, query: str) -> str:
                return "test"

        app = TestApp()
        tru_app = TruApp(
            app, app_name="ManualApp", app_version="v1", main_method=app.query
        )

        # Should not be registered
        metric_names = [
            m["metric"].metric_type for m in tru_app.get_custom_metrics()
        ]
        self.assertNotIn("manual_metric", metric_names)

    def test_explicit_evaluation_config_registration(self) -> None:
        """Test explicit registration with EvaluationConfig."""

        @custom_metric(metric_type="explicit_metric", auto_register=False)
        def explicit_metric(query: str, output: str) -> float:
            return 0.6

        eval_config = (
            EvaluationConfig(
                metric_type="explicit_metric", computation_type="client"
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

        class TestApp:
            @instrument()
            def query(self, query: str) -> str:
                return "test output"

        app = TestApp()
        tru_app = TruApp(
            app, app_name="ExplicitApp", app_version="v1", main_method=app.query
        )

        # Explicitly register metric
        tru_app.add_metric_with_evaluation_config(explicit_metric, eval_config)

        # Verify registration
        custom_metrics = tru_app.get_custom_metrics()
        self.assertEqual(len(custom_metrics), 1)

        metric_info = custom_metrics[0]
        self.assertEqual(metric_info["metric"].metric_type, "explicit_metric")
        self.assertEqual(metric_info["computation_type"], "client")
        self.assertIn("evaluation_config", metric_info)

    def test_multi_parameter_function_requires_config(self) -> None:
        """Test that multi-parameter functions require explicit EvaluationConfig."""

        @custom_metric(metric_type="multi_param", auto_register=True)
        def multi_param_metric(query: str, output: str) -> float:
            return 0.5

        class TestApp:
            @instrument()
            def query(self, query: str) -> str:
                return "test"

        app = TestApp()
        # This should not auto-register the multi-parameter function
        tru_app = TruApp(
            app, app_name="MultiApp", app_version="v1", main_method=app.query
        )

        # Should be in pending but not registered
        metric_names = [
            m["metric"].metric_type for m in tru_app.get_custom_metrics()
        ]
        self.assertNotIn("multi_param", metric_names)
        self.assertTrue(
            any(m[0].metric_type == "multi_param" for m in _pending_metrics)
        )

    def test_multiple_apps_registration(self) -> None:
        """Test that metrics register with multiple TruApps."""

        class TestApp:
            @instrument()
            def query(self, query: str) -> str:
                return "test"

        # Create two TruApps
        app1 = TestApp()
        tru_app1 = TruApp(
            app1, app_name="App1", app_version="v1", main_method=app1.query
        )

        app2 = TestApp()
        tru_app2 = TruApp(
            app2, app_name="App2", app_version="v1", main_method=app2.query
        )

        # Define metric after both apps exist
        @custom_metric(metric_type="shared_metric", auto_register=True)
        def shared_metric(query: str) -> float:
            return 0.3

        # Should be registered with both apps
        app1_metrics = [
            m["metric"].metric_type for m in tru_app1.get_custom_metrics()
        ]
        app2_metrics = [
            m["metric"].metric_type for m in tru_app2.get_custom_metrics()
        ]

        self.assertIn("shared_metric", app1_metrics)
        self.assertIn("shared_metric", app2_metrics)

    def test_metric_with_evaluation_config_decorator(self) -> None:
        """Test @custom_metric with explicit EvaluationConfig."""
        eval_config = EvaluationConfig(
            metric_type="config_metric", computation_type="client"
        ).add_selector(
            "input_text",
            Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        )

        @custom_metric(evaluation_config=eval_config, auto_register=True)
        def config_metric(input_text: str) -> float:
            return len(input_text) / 50.0

        class TestApp:
            @instrument()
            def process(self, text: str) -> str:
                return f"Processed: {text}"

        app = TestApp()
        tru_app = TruApp(
            app, app_name="ConfigApp", app_version="v1", main_method=app.process
        )

        # Should be auto-registered with the explicit config
        custom_metrics = tru_app.get_custom_metrics()
        self.assertEqual(len(custom_metrics), 1)

        metric_info = custom_metrics[0]
        self.assertEqual(metric_info["metric"].metric_type, "config_metric")
        self.assertIn("evaluation_config", metric_info)
        self.assertEqual(
            metric_info["evaluation_config"].metric_type, "config_metric"
        )
