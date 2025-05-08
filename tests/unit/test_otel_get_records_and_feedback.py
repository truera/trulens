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
from trulens.core.schema.app import AppDefinition
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
        self.app_id = AppDefinition._compute_app_id(
            self.app_name, self.app_version
        )

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
        self.STATIC_APP_ID = AppDefinition._compute_app_id(
            self.STATIC_APP_NAME, self.STATIC_APP_VERSION
        )
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
            self.assertEqual(row["num_events"], expected_num_events)

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

    def _create_event_from_template(
        self,
        event_id: str,
        record_id: str,
        app_name: str,
        app_version: str,
        app_id: str,
        start_timestamp: str,
        timestamp: str,
        span_type: str = "record_root",
        query: str = "What is the best coffee?",
        return_value: List[str] = None,
    ) -> Event:
        """Create an event by filling in a template span file.

        Args:
            event_id: Unique identifier for the event
            record_id: Record ID associated with the event
            app_name: Name of the app
            app_version: Version of the app
            app_id: ID of the app
            start_timestamp: Start timestamp of the event
            timestamp: End timestamp of the event
            span_type: Type of span (default: "record_root")
            query: Query text (default: "What is the best coffee?")
            return_value: List of return values (default: None)

        Returns:
            An Event object with the specified attributes
        """
        if return_value is None:
            return_value = ["Sample response"]

        # Load template file
        template_path = (
            Path(__file__).parent
            / "data"
            / "test_otel_spans"
            / "template_record_root_span.json"
        )
        with open(template_path, "r") as f:
            template = f.read()

        # Fill in template values
        event_data = (
            template.replace("{{event_id}}", event_id)
            .replace("{{app_name}}", app_name)
            .replace("{{app_version}}", app_version)
            .replace("{{app_id}}", app_id)
            .replace("{{query}}", query)
            .replace("{{return_value}}", json.dumps(return_value))
            .replace("{{record_id}}", record_id)
            .replace("{{span_type}}", span_type)
            .replace("{{start_timestamp}}", start_timestamp)
            .replace("{{timestamp}}", timestamp)
            .replace("{{span_id}}", f"span_{record_id}")
        )

        event = Event.model_validate(json.loads(event_data))
        return event

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

            # Calculate expected latency in seconds
            start_time = datetime.fromisoformat(self.STATIC_START_TIME)
            end_time = datetime.fromisoformat(self.STATIC_END_TIME)
            expected_latency = (end_time - start_time).total_seconds()
            self.assertEqual(row["latency"], expected_latency)

            # Verify JSON fields
            self._verify_json_fields(
                row,
                self.STATIC_APP_ID,
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
            app,
            app_name=self.app_name,
            app_version=self.app_version,
            app_id=self.app_id,
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
            self.assertEqual(row["app_id"], self.app_id)

            # Verify JSON fields
            self._verify_json_fields(
                row, self.app_id, self.GEN_QUESTION, self.GEN_ANSWER
            )

    def test_get_records_and_feedback_otel_gen_eval_spans(self):
        """Test that _get_records_and_feedback_otel correctly processes eval spans generated by an app.

        This test creates a simple app with record root, retrieval, generation, and eval spans,
        invokes it, and then verifies that the method correctly processes the spans.
        """
        # Create app
        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name=self.app_name,
            app_version=self.app_version,
            app_id=self.app_id,
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
            self.assertEqual(row["app_id"], self.app_id)

            # Verify JSON fields
            self._verify_json_fields(
                row, self.app_id, self.GEN_QUESTION, self.GEN_ANSWER
            )

            # Verify feedback columns
            self._verify_feedback_columns(
                feedback_cols,
                df,
                self.GEN_FEEDBACK_NAME,
                self.GEN_NUM_CALLS,
                self.GEN_COST_CURRENCY,
            )

    def test_get_paginated_record_ids_otel(self):
        """Test that _get_paginated_record_ids_otel correctly paginates and filters record IDs."""
        # Load example spans from test_otel_spans
        event_files = [
            "record_root_span.json",
            "generation_span.json",
            "retrieval_span.json",
        ]
        events = self._load_events_from_json(event_files)

        # Create a second app with multiple records
        second_app_name = "second_app"
        second_app_version = "1.0.0"
        second_record_ids = [
            "second_record_1",
            "second_record_2",
            "second_record_3",
        ]

        # Create events for the second app using the template
        for i in range(len(second_record_ids)):
            event = self._create_event_from_template(
                event_id=f"event_{second_record_ids[i]}",
                record_id=second_record_ids[i],
                app_name=second_app_name,
                app_version=second_app_version,
                app_id=AppDefinition._compute_app_id(
                    second_app_name, second_app_version
                ),
                start_timestamp=f"2025-04-11 10:19:5{i}.993997",
                timestamp="2025-04-11 10:20:30.50",
            )
            events.append(event)

        # Insert all Events into the database
        for event in events:
            self.db.insert_event(event)

        with self.db.session.begin() as session:
            # Test without pagination or filtering - should get all record IDs
            stmt = self.db._get_paginated_record_ids_otel()
            results = session.execute(stmt).all()
            self.assertEqual(
                len(results), 4
            )  # 1 from first app + 3 from second app
            # Verify all record IDs are present
            record_ids = {r.record_id for r in results}
            self.assertEqual(
                record_ids, {self.STATIC_RECORD_ID, *second_record_ids}
            )

            # Test with first app_name filtering
            stmt = self.db._get_paginated_record_ids_otel(
                app_name=self.STATIC_APP_NAME
            )
            results = session.execute(stmt).all()
            self.assertEqual(
                len(results), 1
            )  # Should get only first app's record
            self.assertEqual(results[0].record_id, self.STATIC_RECORD_ID)

            # Test with second app_name filtering
            stmt = self.db._get_paginated_record_ids_otel(
                app_name=second_app_name
            )
            results = session.execute(stmt).all()
            self.assertEqual(
                len(results), 3
            )  # Should get all second app's records
            record_ids = {r.record_id for r in results}
            self.assertEqual(record_ids, set(second_record_ids))

            # Test with non-matching app_name
            stmt = self.db._get_paginated_record_ids_otel(
                app_name="non_existent_app"
            )
            results = session.execute(stmt).all()
            self.assertEqual(len(results), 0)  # Should get no results

            # Test with limit
            stmt = self.db._get_paginated_record_ids_otel(limit=2)
            results = session.execute(stmt).all()
            self.assertEqual(len(results), 2)  # Should respect limit

            # Test with offset
            stmt = self.db._get_paginated_record_ids_otel(offset=2)
            results = session.execute(stmt).all()
            self.assertEqual(len(results), 2)  # Should skip first 2 records

            # Test with both limit and offset
            stmt = self.db._get_paginated_record_ids_otel(limit=2, offset=1)
            results = session.execute(stmt).all()
            self.assertEqual(
                len(results), 2
            )  # Should get 2 records after skipping 1

            # Test ordering by timestamp
            stmt = self.db._get_paginated_record_ids_otel()
            results = session.execute(stmt).all()
            # Verify that results are ordered by start_timestamp in ascending order
            timestamps = [r.min_start_timestamp for r in results]
            self.assertEqual(timestamps, sorted(timestamps))

    def test_get_paginated_record_ids_otel_without_record_id(self):
        """Test that _get_paginated_record_ids_otel handles events without record IDs correctly."""
        # Create an event without a record_id
        empty_record_id_event = self._create_event_from_template(
            event_id="event_without_record_id",
            record_id="",  # Empty record_id
            app_name=self.app_name,
            app_version=self.app_version,
            app_id=self.app_id,
            start_timestamp="2025-04-11 10:19:55.993997",
            timestamp="2025-04-11 10:19:56.356310",
        )
        self.db.insert_event(empty_record_id_event)

        # Create a normal event with a record_id
        normal_event = self._create_event_from_template(
            event_id="normal_event",
            record_id="normal_record_id",
            app_name=self.app_name,
            app_version=self.app_version,
            app_id=self.app_id,
            start_timestamp="2025-04-11 10:19:55.993997",
            timestamp="2025-04-11 10:19:56.356310",
        )
        self.db.insert_event(normal_event)

        with self.db.session.begin() as session:
            stmt = self.db._get_paginated_record_ids_otel()
            results = session.execute(stmt).all()

            # Should only return the event with a valid record_id
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].record_id, "normal_record_id")

    def test_get_records_and_feedback_otel_pagination_with_app_ids(self):
        """Test that _get_records_and_feedback_otel correctly handles pagination with app_id filtering.

        This test verifies that when filtering by app_ids after retrieving records,
        we may get fewer records than requested by the limit parameter.
        """
        # Create multiple apps with different app_ids
        app1_name = "app1"
        app1_version = "1.0.0"
        app2_name = "app2"
        app2_version = "1.0.0"

        # Create events for app1
        app1_events = []
        for i in range(3):  # Create 3 records for app1
            event = self._create_event_from_template(
                event_id=f"app1_event_{i}",
                record_id=f"app1_record_{i}",
                app_name=app1_name,
                app_version=app1_version,
                app_id=AppDefinition._compute_app_id(app1_name, app1_version),
                start_timestamp=f"2025-04-11 10:19:5{i}.993997",
                timestamp="2025-04-11 10:20:30.50",
            )
            # Ensure app name and version are correctly set in record_attributes
            event.record_attributes["ai.observability.app_name"] = app1_name
            event.record_attributes["ai.observability.app_version"] = (
                app1_version
            )
            app1_events.append(event)

        # Create events for app2
        app2_events = []
        for i in range(3):  # Create 3 records for app2
            event = self._create_event_from_template(
                event_id=f"app2_event_{i}",
                record_id=f"app2_record_{i}",
                app_name=app2_name,
                app_version=app2_version,
                app_id=AppDefinition._compute_app_id(app2_name, app2_version),
                start_timestamp=f"2025-04-11 10:19:5{i + 3}.993997",
                timestamp="2025-04-11 10:20:30.50",
            )
            # Ensure app name and version are correctly set in record_attributes
            event.record_attributes["ai.observability.app_name"] = app2_name
            event.record_attributes["ai.observability.app_version"] = (
                app2_version
            )
            app2_events.append(event)

        # Insert all events
        for event in app1_events + app2_events:
            self.db.insert_event(event)

        # Compute app_ids
        app1_id = AppDefinition._compute_app_id(app1_name, app1_version)
        app2_id = AppDefinition._compute_app_id(app2_name, app2_version)

        # Test 1: Get all records without filtering
        df, _ = self.db._get_records_and_feedback_otel(limit=10)
        self.assertEqual(len(df), 6)  # Should get 4 records as requested

        # Test 2: Filter by app1_id with limit=4
        df, _ = self.db._get_records_and_feedback_otel(
            app_ids=[app1_id], limit=4
        )
        self.assertEqual(
            len(df), 3
        )  # Should get only 3 records (all from app1)

        # Test 3: Filter by app2_id with limit=4
        df, _ = self.db._get_records_and_feedback_otel(
            app_ids=[app2_id], limit=4
        )
        self.assertEqual(
            len(df), 3
        )  # Should get only 3 records (all from app2)

        # Test 4: Filter by both app_ids with limit=4
        df, _ = self.db._get_records_and_feedback_otel(
            app_ids=[app1_id, app2_id], limit=4
        )
        self.assertEqual(
            len(df), 4
        )  # Should get 4 records (3 from app1, 1 from app2)

        # Test 5: Filter by non-existent app_id
        df, _ = self.db._get_records_and_feedback_otel(
            app_ids=["non_existent_app_id"], limit=4
        )
        self.assertEqual(len(df), 0)  # Should get 0 records

        # Test 6: Verify ordering is maintained
        df, _ = self.db._get_records_and_feedback_otel(limit=6)
        timestamps = df["ts"].tolist()
        self.assertEqual(
            timestamps, sorted(timestamps)
        )  # Should be ordered by timestamp

        # Test 7: Verify offset works correctly with app_id filtering
        df, _ = self.db._get_records_and_feedback_otel(
            app_ids=[app1_id], limit=2, offset=1
        )
        self.assertEqual(len(df), 2)  # Should get 2 records from app1
        self.assertEqual(
            df.iloc[0]["record_id"], "app1_record_1"
        )  # Should skip first record
        self.assertEqual(
            df.iloc[1]["record_id"], "app1_record_2"
        )  # Should get second and third records
