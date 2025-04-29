"""
Tests for the _get_records_and_feedback_otel method in SQLAlchemyDB.
"""

from datetime import datetime
import json
from pathlib import Path

import pandas as pd
from trulens.core.schema.event import Event
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase


class TestOtelGetRecordsAndFeedback(OtelTestCase):
    """Test the _get_records_and_feedback_otel method in SQLAlchemyDB."""

    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # OTEL tracing is enabled by via OtelTestCase

        # Create a TruSession and reset the database
        self.session = TruSession()
        self.session.reset_database()
        self.db = self.session.connector.db

        # Set app name and version
        self.app_name = "test_app"
        self.app_version = "1.0.0"

    # # TODO: IMPLEMENT THIS
    # def _create_record_root_span(
    #     self, record_id, trace_id, input_text, output_text, token_count=10
    # ):
    #     return None

    # # TODO: IMPLEMENT THIS
    # def _create_retrieval_span(
    #     self, record_id, trace_id, input_text, output_text, token_count=10
    # ):
    #     return None

    # # TODO: IMPLEMENT THIS
    # def _create_generation_span(
    #     self, record_id, trace_id, input_text, output_text, token_count=10
    # ):
    #     return None

    # # TODO: IMPLEMENT THIS
    # def _create_rag_flow(
    #     self, record_id=None, trace_id=None, input_text=None, output_text=None
    # ):
    #     """Create a simple RAG flow with three events: RECORD_ROOT, RETRIEVAL, and GENERATION.

    #     Args:
    #         TODO: Implement this
    #     Returns:
    #         A tuple containing (record_id, trace_id, root_span_id, retrieval_span_id, generation_span_id)
    #     """

    #     # Create the record root span
    #     record_root_span = self._create_record_root_span()

    #     self.db.insert_event(record_root_span)

    #     # Create the retrieval span
    #     retrieval_span = self._create_retrieval_span()

    #     # Insert the retrieval span into the database
    #     self.db.insert_event(retrieval_span)

    #     # Create the generation span
    #     generation_span = self._create_generation_span(
    #         #
    #     )

    #     # Insert the generation span into the database
    #     self.db.insert_event(generation_span)

    #     return (
    #         record_id,
    #         trace_id,
    #         root_span_id,
    #         retrieval_span_id,
    #         generation_span_id,
    #     )

    def test_get_records_and_feedback_otel_empty(self):
        """Test _get_records_and_feedback_otel with an empty database."""
        # Call the method directly
        records_df, feedback_col_names = (
            self.db._get_records_and_feedback_otel()
        )

        # Check that the result is an empty DataFrame with the expected columns
        self.assertIsInstance(records_df, pd.DataFrame)
        self.assertEqual(len(records_df), 0)
        self.assertEqual(feedback_col_names, [])

    def test_get_records_and_feedback_otel_with_example_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes example spans.

        This test uses the example spans from the JSON files to verify that the method
        correctly processes the spans and produces the expected output.
        """
        # Load the example spans from the JSON files
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "record_root_span.json", "r") as f:
            record_root_span = json.load(f)
        with open(data_dir / "generation_span.json", "r") as f:
            generation_span = json.load(f)
        with open(data_dir / "retrieval_span.json", "r") as f:
            retrieval_span = json.load(f)

        # Convert the spans to Event objects
        record_root_event = Event.model_validate(record_root_span)
        generation_event = Event.model_validate(generation_span)
        retrieval_event = Event.model_validate(retrieval_span)

        # Insert the events into the database
        self.db.insert_event(record_root_event)
        self.db.insert_event(generation_event)
        self.db.insert_event(retrieval_event)

        # Call the method to get the records and feedback
        df, feedback_cols = self.db._get_records_and_feedback_otel()

        # Verify there are no feedback columns (this is a simple rag case)
        self.assertEqual(len(feedback_cols), 0)

        # Verify that the dataframe has the expected structure
        self.assertGreater(len(df), 0, "Dataframe should not be empty")

        # Verify that the dataframe is appropriately grouping the spans by record_id
        # In this case, the 3 spans share the same record_id
        self.assertEqual(len(df), 1)

        # Get the first row of the dataframe
        row = df.iloc[0]

        # Verify the basic record information
        self.assertEqual(row["app_name"], "coffee_rag")
        self.assertEqual(row["app_version"], "openai")
        self.assertEqual(
            row["record_id"], "dd121e81-ccc2-449f-a3ba-e12467d0d671"
        )
        self.assertEqual(row["input"], "What is the best coffee?")
        self.assertTrue(row["output"].startswith("Hello! I'm happy to help."))
        self.assertEqual(row["tags"], "")

        # Verify that the dataframe is appropriately grouping the spans by record_id
        # In this case, the 3 spans share the same record_id
        self.assertEqual(len(row["events"]), 3)

        # Verify that total_tokens and total_cost are the sum from all spans
        retrieval_tokens = retrieval_event.record_attributes[
            "ai.observability.cost.num_tokens"
        ]
        generation_tokens = generation_event.record_attributes[
            "ai.observability.cost.num_tokens"
        ]
        expected_total_tokens = retrieval_tokens + generation_tokens
        self.assertEqual(
            row["total_tokens"],
            expected_total_tokens,
            "Total tokens should be sum of tokens from all spans",
        )

        retrieval_cost = retrieval_event.record_attributes[
            "ai.observability.cost.cost"
        ]
        generation_cost = generation_event.record_attributes[
            "ai.observability.cost.cost"
        ]
        expected_total_cost = retrieval_cost + generation_cost
        self.assertEqual(
            row["total_cost"],
            expected_total_cost,
            msg="Total cost should be sum of costs from all spans",
        )

        # Verify the timestamp and latency
        self.assertEqual(
            row["ts"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "2025-04-11 10:19:56.356391",
        )

        # Calculate expected latency in milliseconds
        start_time = datetime.fromisoformat("2025-04-11 10:19:56.356391")
        end_time = datetime.fromisoformat("2025-04-11 10:19:58.166073")
        expected_latency = (end_time - start_time).total_seconds() * 1000
        self.assertEqual(row["latency"], expected_latency)

        # Verify that the app_id is correctly computed
        expected_app_id = self.db._compute_app_id_otel("coffee_rag", "openai")
        self.assertEqual(row["app_id"], expected_app_id)

        # Verify that the record_json, cost_json, and perf_json are correctly created
        self.assertIsNotNone(row["record_json"])
        self.assertIsNotNone(row["cost_json"])
        self.assertIsNotNone(row["perf_json"])

        # Verify that the record_json contains the expected fields
        record_json = row["record_json"]
        self.assertEqual(
            record_json["record_id"], "dd121e81-ccc2-449f-a3ba-e12467d0d671"
        )
        self.assertEqual(record_json["app_id"], expected_app_id)
        self.assertEqual(record_json["input"], "What is the best coffee?")
        self.assertTrue(
            record_json["output"].startswith("Hello! I'm happy to help.")
        )

        # Verify that the cost_json contains the expected fields
        cost_json = row["cost_json"]
        self.assertEqual(cost_json["n_tokens"], expected_total_tokens)
        self.assertEqual(cost_json["cost"], expected_total_cost)

        # Verify that the perf_json contains the expected fields
        perf_json = row["perf_json"]
        self.assertEqual(
            perf_json["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "2025-04-11 10:19:56",
        )
        self.assertEqual(
            perf_json["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "2025-04-11 10:19:58",
        )
