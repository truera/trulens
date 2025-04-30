"""
Tests for the _get_records_and_feedback_otel method in SQLAlchemyDB.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List

import pandas as pd
from trulens.apps.app import TruApp
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.otel.instrument import instrument
from trulens.core.schema.event import Event
from trulens.core.session import TruSession
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.mock_otel_feedback_computation import (
    all_retrieval_span_attributes,
)
from tests.util.mock_otel_feedback_computation import feedback_function
from tests.util.otel_test_case import OtelTestCase

try:
    import tests.unit.test_otel_feedback_computation
except ImportError:
    pass


class _TestApp:
    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def query(self, question: str) -> str:
        # Call retrieval and generation methods
        self.retrieve(question)
        response = self.generate(question)
        return response

    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
    def retrieve(self, question: str) -> List[str]:
        # Simulate retrieval of documents
        return ["Document 1", "Document 2"]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, question: str) -> str:
        # Simulate generation of response
        return f"Answer to: {question}"


class TestOtelGetRecordsAndFeedback(OtelTestCase):
    """Test the _get_records_and_feedback_otel method in SQLAlchemyDB."""

    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # OTEL tracing is enabled by via OtelTestCase

        # Create a TruSession and reset the database
        self.session = TruSession()
        self.session.reset_database()

        # Ensure connector is properly initialized
        if self.session.connector is None:
            raise RuntimeError("Session connector is None")

        self.db: SQLAlchemyDB = self.session.connector.db

        # Set app name and version
        self.app_name = "test_app"
        self.app_version = "1.0.0"

        # Define common test constants
        # Constants for static RAG test
        self.STATIC_START_TIME = "2025-04-11 10:19:55.993955"
        self.STATIC_END_TIME = "2025-04-11 10:19:58.166175"
        self.STATIC_RECORD_ID = "dd121e81-ccc2-449f-a3ba-e12467d0d671"
        self.STATIC_QUESTION = "What is the best coffee?"
        self.STATIC_ANSWER = (
            "Hello! I'm happy to help. \n\nWhen it comes to the best coffee"
        )
        self.STATIC_APP_NAME = "coffee_rag"
        self.STATIC_APP_VERSION = "openai"
        self.STATIC_NUM_EVENTS = 3

        # Constants for static eval test
        self.STATIC_FEEDBACK_NAME = "answer_relevance"
        # NOTE: there are 3 total calls for the feedback column, see:
        # test_otel_spans/{eval_root_span.json, eval_span_1.json, eval_span_2.json}
        self.STATIC_NUM_CALLS = 3
        self.STATIC_COST_CURRENCY = "USD"

        # Constants for generated tests
        self.GEN_QUESTION = "What is the capital of France?"
        self.GEN_ANSWER = "Answer to: What is the capital of France?"
        self.GEN_FEEDBACK_NAME = "groundedness"
        # NOTE: there are 3 calls: see tests.util.mock_otel_feedback_computation.feedback_function
        self.GEN_NUM_CALLS = 3
        self.GEN_COST_CURRENCY = "USD"

    # Helper methods
    def _verify_dataframe_structure(self, df, expected_num_rows=1):
        """Verify the basic structure of the dataframe.

        Args:
            df: The dataframe to verify
            expected_num_rows: Expected number of rows in the dataframe

        Returns:
            The first row of the dataframe if expected_num_rows > 0, otherwise None
        """
        # Verify that the dataframe is not empty if expected_num_rows > 0
        if expected_num_rows > 0:
            self.assertGreater(len(df), 0, "Dataframe should not be empty")
        else:
            self.assertEqual(len(df), 0, "Dataframe should be empty")

        # Verify that the dataframe is appropriately grouping the spans by record_id
        self.assertEqual(len(df), expected_num_rows)

        return df.iloc[0] if expected_num_rows > 0 and len(df) > 0 else None

    def _verify_record_information(
        self,
        row,
        app_name,
        app_version,
        input_text,
        output_text,
        expected_num_events=None,
    ):
        """Verify the basic record information in a row.

        Args:
            row: The row to verify (can be None)
            app_name: Expected app name
            app_version: Expected app version
            input_text: Expected input text
            output_text: Expected output text
            expected_num_events: Expected number of events (if None, not verified)
        """
        if row is None:
            return

        # Verify the basic record information
        self.assertEqual(row["app_name"], app_name)
        self.assertEqual(row["app_version"], app_version)
        # (hacky): uses startswith to circumvent typing out long inputs/outputs
        self.assertTrue(row["input"].startswith(input_text))
        self.assertTrue(row["output"].startswith(output_text))

        # Verify that the dataframe is appropriately grouping the spans by record_id
        if expected_num_events is not None:
            self.assertEqual(len(row["events"]), expected_num_events)

    def _verify_json_fields(
        self, row, app_id, input_text, output_text, record_id=None
    ):
        """Verify the JSON fields in a row.

        Args:
            row: The row to verify (can be None)
            app_id: Expected app ID
            input_text: Expected input text
            output_text: Expected output text
            record_id: Expected record ID (if None, not verified)
        """
        if row is None:
            return

        # Verify that the record_json, cost_json, and perf_json are correctly created
        self.assertIsNotNone(row["record_json"])
        self.assertIsNotNone(row["cost_json"])
        self.assertIsNotNone(row["perf_json"])

        # Verify that the record_json contains the expected fields
        record_json = row["record_json"]
        self.assertEqual(record_json["app_id"], app_id)
        # Use startswith to handle potential variations in the text
        self.assertTrue(record_json["input"].startswith(input_text))
        self.assertTrue(record_json["output"].startswith(output_text))

        if record_id is not None:
            self.assertEqual(record_json["record_id"], record_id)

    def _verify_feedback_columns(
        self,
        feedback_cols,
        df,
        feedback_name,
        expected_num_calls,
        cost_currency="USD",
    ):
        """Verify the feedback columns in the dataframe.

        Args:
            feedback_cols: The feedback column names
            df: The dataframe to verify
            feedback_name: Expected feedback name
            expected_num_calls: Expected number of calls
            cost_currency: Expected cost currency
        """
        # Verify feedback columns are properly created
        self.assertEqual(len(feedback_cols), 1)
        self.assertEqual(feedback_cols[0], feedback_name)

        # Assert that all expected feedback columns exist
        self.assertIn(f"{feedback_name}_calls", df.columns)
        self.assertEqual(
            len(df[f"{feedback_name}_calls"][0]), expected_num_calls
        )
        self.assertIn(
            f"{feedback_name} feedback cost in {cost_currency}",
            df.columns,
        )
        self.assertIn(f"{feedback_name} direction", df.columns)

    def _load_events_from_json(self, event_files):
        """Load events from JSON files.

        Args:
            event_files: List of event file names to load

        Returns:
            List of Event objects
        """
        data_dir = Path(__file__).parent / "data" / "test_otel_spans"
        events = []

        for file_name in event_files:
            with open(data_dir / file_name, "r") as f:
                event_data = json.load(f)
                event = Event.model_validate(event_data)
                events.append(event)

        return events

    # Tests
    def test_get_records_and_feedback_otel_empty(self):
        """Test _get_records_and_feedback_otel with an empty database."""
        records_df, feedback_col_names = (
            self.db._get_records_and_feedback_otel()
        )

        self.assertIsInstance(records_df, pd.DataFrame)
        self.assertEqual(len(records_df), 0)
        self.assertEqual(feedback_col_names, [])

    def test_get_records_and_feedback_otel_static_rag_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes example RAG spans.

        This test uses example RAG spans from JSON files in tests/unit/data/test_otel_spans
        to verify the derived records dataframe and feedback columns are created correctly.
        """
        # Load example spans from test_otel_spans
        event_files = [
            "record_root_span.json",
            "generation_span.json",
            "retrieval_span.json",
        ]
        events = self._load_events_from_json(event_files)

        # Insert Events into the database
        for event in events:
            self.db.insert_event(event)

        df, feedback_cols = self.db._get_records_and_feedback_otel()

        # Verify there are no feedback columns (this is a non-EVAL RAG scenario)
        self.assertEqual(len(feedback_cols), 0)

        # Verify dataframe structure
        row = self._verify_dataframe_structure(df)

        # Verify record information
        self._verify_record_information(
            row,
            self.STATIC_APP_NAME,
            self.STATIC_APP_VERSION,
            self.STATIC_QUESTION,
            self.STATIC_ANSWER,
            self.STATIC_NUM_EVENTS,
        )

        # Verify that total_tokens and total_cost are the sum from all spans
        if row is not None:
            # Find retrieval and generation spans by their span_type
            retrieval_event = next(
                event
                for event in events
                if event.record_attributes.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RETRIEVAL
            )
            generation_event = next(
                event
                for event in events
                if event.record_attributes.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.GENERATION
            )

            retrieval_tokens = retrieval_event.record_attributes[
                SpanAttributes.COST.NUM_TOKENS
            ]
            generation_tokens = generation_event.record_attributes[
                SpanAttributes.COST.NUM_TOKENS
            ]
            expected_total_tokens = retrieval_tokens + generation_tokens
            self.assertEqual(
                row["total_tokens"],
                expected_total_tokens,
                "Total tokens should be sum of tokens from all spans",
            )

            retrieval_cost = retrieval_event.record_attributes[
                SpanAttributes.COST.COST
            ]
            generation_cost = generation_event.record_attributes[
                SpanAttributes.COST.COST
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
                self.STATIC_START_TIME,
            )

            # Calculate expected latency in milliseconds
            start_time = datetime.fromisoformat(self.STATIC_START_TIME)
            end_time = datetime.fromisoformat(self.STATIC_END_TIME)
            expected_latency = (end_time - start_time).total_seconds() * 1000
            self.assertEqual(row["latency"], expected_latency)

            # Verify that the app_id is correctly computed
            expected_app_id = self.db._compute_app_id_otel(
                self.STATIC_APP_NAME, self.STATIC_APP_VERSION
            )
            self.assertEqual(row["app_id"], expected_app_id)

            # Verify JSON fields
            self._verify_json_fields(
                row,
                expected_app_id,
                self.STATIC_QUESTION,
                self.STATIC_ANSWER,
                self.STATIC_RECORD_ID,
            )

            # Verify that the cost_json contains the expected fields
            cost_json = row["cost_json"]
            self.assertEqual(cost_json["n_tokens"], expected_total_tokens)
            self.assertEqual(cost_json["cost"], expected_total_cost)

            # Verify that the perf_json contains the expected fields
            perf_json = row["perf_json"]
            self.assertEqual(
                perf_json["start_time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.STATIC_START_TIME,
            )
            self.assertEqual(
                perf_json["end_time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.STATIC_END_TIME,
            )

    def test_get_records_and_feedback_otel_static_eval_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes eval spans.

        This test uses example RAG and eval spans from JSON files in tests/unit/data/test_otel_spans
        to verify the derived records dataframe and feedback columns are created correctly.
        """
        # Load example spans from test_otel_spans
        event_files = [
            "record_root_span.json",
            "generation_span.json",
            "retrieval_span.json",
            "eval_root_span.json",
            "eval_span_1.json",
            "eval_span_2.json",
        ]
        events = self._load_events_from_json(event_files)

        # Insert Events into the database
        for event in events:
            self.db.insert_event(event)

        df, feedback_cols = self.db._get_records_and_feedback_otel()

        # Verify feedback columns
        self._verify_feedback_columns(
            feedback_cols,
            df,
            self.STATIC_FEEDBACK_NAME,
            self.STATIC_NUM_CALLS,
            self.STATIC_COST_CURRENCY,
        )

    def test_get_records_and_feedback_otel_gen_rag_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes RAG spans generated by an app.

        This test creates a simple app with record root, retrieval, and generation spans,
        invokes it, and then verifies that the method correctly processes the spans.
        """
        # Create app
        app = _TestApp()
        tru_app = TruApp(
            app, app_name=self.app_name, app_version=self.app_version
        )

        # Record and invoke
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_kwargs={"question": self.GEN_QUESTION},
        )

        # Force flush to ensure all spans are written to the database
        self.session.force_flush()

        # Get records and feedback
        df, feedback_cols = self.db._get_records_and_feedback_otel()

        # Verify there are no feedback columns (this is a non-EVAL RAG scenario)
        self.assertEqual(len(feedback_cols), 0)

        # Verify dataframe structure
        row = self._verify_dataframe_structure(df)

        # Verify record information
        self._verify_record_information(
            row,
            self.app_name,
            self.app_version,
            self.GEN_QUESTION,
            self.GEN_ANSWER,
            self.STATIC_NUM_EVENTS,
        )

        # Verify that the app_id is correctly computed
        if row is not None:
            expected_app_id = self.db._compute_app_id_otel(
                self.app_name, self.app_version
            )
            self.assertEqual(row["app_id"], expected_app_id)

            # Verify JSON fields
            self._verify_json_fields(
                row, expected_app_id, self.GEN_QUESTION, self.GEN_ANSWER
            )

    def test_get_records_and_feedback_otel_gen_eval_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes eval spans generated by an app.

        This test creates a simple app with record root, retrieval, generation, and eval spans,
        invokes it, and then verifies that the method correctly processes the spans.
        """
        # Create app
        app = _TestApp()
        tru_app = TruApp(
            app, app_name=self.app_name, app_version=self.app_version
        )

        # Record and invoke the query method
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_kwargs={"question": self.GEN_QUESTION},
        )
        self.session.force_flush()

        events = self._get_events()
        spans = tests.unit.test_otel_feedback_computation._convert_events_to_MinimalSpanInfos(
            events
        )
        record_root = RecordGraphNode.build_graph(spans)
        _compute_feedback(
            record_root,
            self.GEN_FEEDBACK_NAME,
            feedback_function,
            all_retrieval_span_attributes,
        )
        self.session.force_flush()

        # Get records and feedback
        df, feedback_cols = self.db._get_records_and_feedback_otel()

        # Verify dataframe structure
        row = self._verify_dataframe_structure(df)

        # Verify record information
        self._verify_record_information(
            row,
            self.app_name,
            self.app_version,
            self.GEN_QUESTION,
            self.GEN_ANSWER,
        )

        # Verify that the app_id is correctly computed
        if row is not None:
            expected_app_id = self.db._compute_app_id_otel(
                self.app_name, self.app_version
            )
            self.assertEqual(row["app_id"], expected_app_id)

            # Verify JSON fields
            self._verify_json_fields(
                row, expected_app_id, self.GEN_QUESTION, self.GEN_ANSWER
            )

            # Verify feedback columns
            self._verify_feedback_columns(
                feedback_cols,
                df,
                self.GEN_FEEDBACK_NAME,
                self.GEN_NUM_CALLS,
                self.GEN_COST_CURRENCY,
            )
