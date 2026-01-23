"""
Tests for virtual run functionality.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd

try:
    from trulens.core.run import Run
    from trulens.otel.semconv.trace import SpanAttributes

    from tests.util.otel_test_case import OtelTestCase
except Exception:
    pass


class TestVirtualRunMethods(OtelTestCase):
    """Test virtual run helper methods with proper OTEL setup"""

    def test_virtual_run_start_signature(self):
        """Test that Run.start no longer accepts virtual parameter"""
        import inspect

        sig = inspect.signature(Run.start)
        self.assertNotIn("virtual", sig.parameters)

    def test_span_type_enum_mapping(self):
        """Test dynamic span type enum mapping"""
        # Create a dummy run instance to test the method
        mock_app = MagicMock()
        mock_app.app_name = "test"
        mock_app.app_version = "1.0"

        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="test",
            tru_session=MagicMock(),
            object_name="TEST",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="test", column_spec={}, source_type="TABLE"
            ),
        )

        # Test known span types
        self.assertEqual(
            run._get_span_type_enum("record_root"),
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        self.assertEqual(
            run._get_span_type_enum("retrieval"),
            SpanAttributes.SpanType.RETRIEVAL,
        )
        self.assertEqual(
            run._get_span_type_enum("generation"),
            SpanAttributes.SpanType.GENERATION,
        )
        self.assertIsNone(run._get_span_type_enum("unknown_type"))

    def test_span_attributes_class_mapping(self):
        """Test dynamic span attributes class mapping"""
        mock_app = MagicMock()
        mock_app.app_name = "test"
        mock_app.app_version = "1.0"

        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="test",
            tru_session=MagicMock(),
            object_name="TEST",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="test", column_spec={}, source_type="TABLE"
            ),
        )

        # Test known span attributes classes
        self.assertEqual(
            run._get_span_attributes_class("record_root"),
            SpanAttributes.RECORD_ROOT,
        )
        self.assertEqual(
            run._get_span_attributes_class("retrieval"),
            SpanAttributes.RETRIEVAL,
        )
        self.assertEqual(
            run._get_span_attributes_class("generation"),
            SpanAttributes.GENERATION,
        )
        self.assertIsNone(run._get_span_attributes_class("unknown_type"))

    @patch("trulens.core.otel.instrument.OtelRecordingContext")
    def test_virtual_spans_integration(self, mock_otel_context):
        """Test that virtual spans can be created without errors"""
        mock_app = MagicMock()
        mock_app.app = None
        mock_app.app_name = "Test Virtual App"
        mock_app.app_version = "1.0"

        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="virtual_main",
            tru_session=MagicMock(),
            object_name="TEST_APP",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test_virtual_run",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="TEST_TABLE",
                column_spec={
                    "record_root.input": "QUERY",
                    "record_root.output": "OUTPUT",
                },
                source_type="TABLE",
            ),
        )

        input_df = pd.DataFrame({
            "QUERY": ["What is AI?"],
            "OUTPUT": ["AI is..."],
        })

        dataset_spec = {
            "record_root.input": "QUERY",
            "record_root.output": "OUTPUT",
        }

        mock_otel_context.return_value.__enter__.return_value = MagicMock()
        mock_otel_context.return_value.__exit__.return_value = None

        # Should not raise any exceptions
        run._create_virtual_spans(input_df, dataset_spec, 1)

        # Verify OtelRecordingContext was called
        self.assertTrue(mock_otel_context.called)

    def test_should_process_as_array(self):
        """Test array attribute detection by constant"""
        mock_app = MagicMock()
        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="test",
            tru_session=MagicMock(),
            object_name="TEST",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="test", column_spec={}, source_type="TABLE"
            ),
        )

        # Test array attributes
        self.assertTrue(
            run._should_process_as_array(
                SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS
            )
        )
        self.assertTrue(
            run._should_process_as_array(
                SpanAttributes.GRAPH_NODE.NODES_EXECUTED
            )
        )
        self.assertTrue(
            run._should_process_as_array(
                SpanAttributes.RERANKER.INPUT_CONTEXT_TEXTS
            )
        )

        # Test non-array attributes
        self.assertFalse(
            run._should_process_as_array(SpanAttributes.RETRIEVAL.QUERY_TEXT)
        )
        self.assertFalse(
            run._should_process_as_array(SpanAttributes.RECORD_ROOT.INPUT)
        )

    def test_should_process_as_array_by_name(self):
        """Test array attribute detection by name"""
        mock_app = MagicMock()
        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="test",
            tru_session=MagicMock(),
            object_name="TEST",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="test", column_spec={}, source_type="TABLE"
            ),
        )

        # Test array attribute names
        self.assertTrue(
            run._should_process_as_array_by_name("retrieved_contexts")
        )
        self.assertTrue(run._should_process_as_array_by_name("nodes_executed"))
        self.assertTrue(
            run._should_process_as_array_by_name("input_context_texts")
        )

        # Test non-array attribute names
        self.assertFalse(run._should_process_as_array_by_name("query_text"))
        self.assertFalse(run._should_process_as_array_by_name("input"))
        self.assertFalse(run._should_process_as_array_by_name("output"))

    def test_process_array_attribute(self):
        """Test array attribute processing with various inputs"""
        mock_app = MagicMock()
        run = Run(
            run_dao=MagicMock(),
            app=mock_app,
            main_method_name="test",
            tru_session=MagicMock(),
            object_name="TEST",
            object_type="EXTERNAL AGENT",
            object_version="1.0",
            run_name="test",
            run_metadata=Run.RunMetadata(),
            source_info=Run.SourceInfo(
                name="test", column_spec={}, source_type="TABLE"
            ),
        )

        # Test comma-separated string
        result = run._process_array_attribute("Tokyo, capital city, Japan")
        self.assertEqual(result, ["Tokyo", "capital city", "Japan"])

        # Test JSON array string
        result = run._process_array_attribute('["doc1", "doc2", "doc3"]')
        self.assertEqual(result, ["doc1", "doc2", "doc3"])

        # Test already a list
        result = run._process_array_attribute(["item1", "item2"])
        self.assertEqual(result, ["item1", "item2"])

        # Test empty/None values
        self.assertEqual(run._process_array_attribute(""), [])
        self.assertEqual(run._process_array_attribute(None), [])

        # Test single value
        result = run._process_array_attribute("single_item")
        self.assertEqual(result, ["single_item"])

        # Test whitespace handling
        result = run._process_array_attribute("  item1  ,  item2  ")
        self.assertEqual(result, ["item1", "item2"])
