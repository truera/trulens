"""
E2E tests for client-side custom metrics with Snowflake connector.
"""

import os
import time
import uuid

import pandas as pd
import pytest
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.snowflake_test_case import SnowflakeTestCase


def text2sql_quality(query: str, sql: str) -> float:
    """Custom metric to evaluate text-to-SQL quality."""
    if "SELECT" in sql.upper() and len(query) > 10:
        return 0.9
    elif "SELECT" in sql.upper():
        return 0.7
    else:
        return 0.3


def custom_accuracy(query: str) -> float:
    """Custom accuracy metric based on query length."""
    return min(len(query) / 100.0, 1.0)


class TestSnowflakeClientSideCustomMetrics(SnowflakeTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_TRULENS_OTEL_TRACING = os.getenv("TRULENS_OTEL_TRACING")
        os.environ["TRULENS_OTEL_TRACING"] = (
            "1"  # Enable OTEL for custom metrics
        )
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._orig_TRULENS_OTEL_TRACING is not None:
            os.environ["TRULENS_OTEL_TRACING"] = cls._orig_TRULENS_OTEL_TRACING
        else:
            del os.environ["TRULENS_OTEL_TRACING"]
        return super().tearDownClass()

    def _create_test_app(self) -> TruApp:
        """Create a test app for custom metrics evaluation."""

        class Text2SQLApp:
            @instrument(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                attributes={
                    SpanAttributes.RECORD_ROOT.INPUT: "query",
                    SpanAttributes.RECORD_ROOT.OUTPUT: "return",
                },
            )
            def generate_sql(self, query: str) -> str:
                if "users" in query.lower():
                    return "SELECT * FROM users"
                elif "orders" in query.lower():
                    return "SELECT * FROM orders"
                else:
                    return "SELECT 1"

        app = Text2SQLApp()
        return TruApp(
            app,
            app_name="Text2SQLApp",
            app_version="v1",
            main_method=app.generate_sql,
            connector=self.get_connector(),
        )

    def _create_metric_configs(self) -> list:
        """Create test metric configurations."""
        text2sql_metric = Metric(
            name="text2sql_evaluation_v1",
            implementation=text2sql_quality,
            metric_type="text2sql",
            selectors={
                "query": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
                "sql": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
                ),
            },
        )

        accuracy_metric = Metric(
            name="query_length_accuracy_v1",
            implementation=custom_accuracy,
            metric_type="accuracy",
            selectors={
                "query": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            },
        )

        return [text2sql_metric, accuracy_metric]

    @pytest.mark.optional
    def test_client_side_custom_metrics_e2e(self) -> None:
        """Test end-to-end client-side custom metrics computation with Snowflake."""

        # Create app and run
        tru_app = self._create_test_app()

        run_name = f"test_custom_metrics_{uuid.uuid4()}"
        run_config = RunConfig(
            run_name=run_name,
            dataset_name="test_queries",
            source_type="DATAFRAME",
            dataset_spec={"input": "query"},
        )

        run = tru_app.add_run(run_config)

        # Create test data
        test_data = pd.DataFrame({
            "query": [
                "Show me all users in the database",
                "Get order information",
                "Simple query",
            ]
        })

        # Start the run
        run.start(input_df=test_data)

        # Wait for completion
        max_wait = 60  # seconds
        start_time = time.time()
        while run.get_status() != "INVOCATION_COMPLETED":
            if time.time() - start_time > max_wait:
                self.fail(f"Run did not complete within {max_wait} seconds")
            time.sleep(2)

        # Create metric configs and compute
        metric_configs = self._create_metric_configs()

        # Test mixed metrics (client-side custom + server-side standard)
        metrics_to_compute = [
            "answer_relevance",  # Server-side metric
            *metric_configs,  # Client-side custom metrics
        ]

        result = run.compute_metrics(metrics_to_compute)
        self.assertIn("Metrics computation in progress", result)

        start_time = time.time()
        while run.get_status() not in ["COMPLETED", "PARTIALLY_COMPLETED"]:
            if time.time() - start_time > max_wait:
                self.fail(
                    f"Metrics computation did not complete within {max_wait} seconds"
                )
            time.sleep(3)

        final_status = run.get_status()
        self.assertIn(final_status, ["COMPLETED", "PARTIALLY_COMPLETED"])

        print(f"✅ E2E test completed successfully with status: {final_status}")

    @pytest.mark.optional
    def test_client_side_metric_validation(self) -> None:
        """Test that metric configs are properly validated."""

        # Test valid metric
        valid_metric = Metric(
            name="valid_test",
            implementation=custom_accuracy,
            metric_type="accuracy",
            selectors={
                "query": Selector(
                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                ),
            },
        )

        # Should have correct name and implementation
        self.assertEqual(valid_metric.name, "valid_test")
        self.assertIsNotNone(valid_metric.imp)
