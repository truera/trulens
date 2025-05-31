"""Test module for dashboard records_utils functions."""

from datetime import datetime
import unittest

import pandas as pd
from trulens.dashboard.utils.records_utils import _filter_duplicate_span_calls


class TestFilterDuplicateSpanCalls(unittest.TestCase):
    """Test cases for _filter_duplicate_span_calls function."""

    def test_filter_duplicate_span_calls_keeps_most_recent_eval_root_id(self):
        """Test that the function keeps only rows from the most recent eval_root_id when args_span_id and args_span_attribute are identical."""
        # Create test data with duplicate args_span_id/args_span_attribute but different eval_root_ids
        test_data = pd.DataFrame([
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.8,
                "other_data": "row1_eval1",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(2023, 1, 1, 11, 0, 0),  # More recent
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.9,
                "other_data": "row1_eval2",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(
                    2023, 1, 1, 11, 5, 0
                ),  # Another row from same eval_root_id
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.85,
                "other_data": "row2_eval2",
            },
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should keep both rows from eval_2 (the most recent eval_root_id)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["eval_root_id"] == "eval_2"))
        self.assertIn("row1_eval2", result["other_data"].values)
        self.assertIn("row2_eval2", result["other_data"].values)

    def test_filter_duplicate_span_calls_keeps_unique_combinations(self):
        """Test that rows with unique args_span_id/args_span_attribute combinations are kept regardless of eval_root_id."""
        test_data = pd.DataFrame([
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.8,
                "other_data": "unique_row_1",
            },
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": {"span": "test_span_2"},  # Different span_id
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.7,
                "other_data": "unique_row_2",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(2023, 1, 1, 11, 0, 0),
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {
                    "attr": "test_attr_2"
                },  # Different attribute
                "score": 0.9,
                "other_data": "unique_row_3",
            },
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should keep all rows since they all have unique combinations
        self.assertEqual(len(result), 3)
        expected_values = {"unique_row_1", "unique_row_2", "unique_row_3"}
        self.assertEqual(set(result["other_data"].values), expected_values)

    def test_filter_duplicate_span_calls_with_none_values(self):
        """Test handling of None values in args_span_id and args_span_attribute."""
        test_data = pd.DataFrame([
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": None,
                "args_span_attribute": None,
                "score": 0.8,
                "other_data": "none_row_1",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(2023, 1, 1, 11, 0, 0),  # More recent
                "args_span_id": None,
                "args_span_attribute": None,
                "score": 0.9,
                "other_data": "none_row_2",
            },
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should keep only the row from the most recent eval_root_id
        self.assertEqual(len(result), 1)
        self.assertEqual(result["eval_root_id"].iloc[0], "eval_2")
        self.assertEqual(result["other_data"].iloc[0], "none_row_2")

    def test_filter_duplicate_span_calls_missing_columns(self):
        """Test that function returns original DataFrame when required columns are missing."""
        test_data = pd.DataFrame([
            {
                "eval_root_id": "eval_1",
                "score": 0.8,
                "other_data": "test_row",
            }
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result, test_data)

    def test_filter_duplicate_span_calls_empty_dataframe(self):
        """Test that function handles empty DataFrame gracefully."""
        test_data = pd.DataFrame()

        result = _filter_duplicate_span_calls(test_data)

        # Should return empty DataFrame
        self.assertTrue(result.empty)

    def test_filter_duplicate_span_calls_complex_scenario(self):
        """Test a complex scenario with multiple groups and eval_root_ids."""
        test_data = pd.DataFrame([
            # Group 1: span_1 + attr_1 (should keep eval_3 rows)
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.8,
                "other_data": "group1_eval1",
            },
            {
                "eval_root_id": "eval_3",
                "timestamp": datetime(
                    2023, 1, 1, 12, 0, 0
                ),  # Most recent for group 1
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.95,
                "other_data": "group1_eval3_row1",
            },
            {
                "eval_root_id": "eval_3",
                "timestamp": datetime(
                    2023, 1, 1, 12, 5, 0
                ),  # Another row from eval_3
                "args_span_id": {"span": "test_span_1"},
                "args_span_attribute": {"attr": "test_attr_1"},
                "score": 0.92,
                "other_data": "group1_eval3_row2",
            },
            # Group 2: span_2 + attr_2 (should keep eval_2 row)
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 30, 0),
                "args_span_id": {"span": "test_span_2"},
                "args_span_attribute": {"attr": "test_attr_2"},
                "score": 0.7,
                "other_data": "group2_eval1",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(
                    2023, 1, 1, 11, 30, 0
                ),  # Most recent for group 2
                "args_span_id": {"span": "test_span_2"},
                "args_span_attribute": {"attr": "test_attr_2"},
                "score": 0.85,
                "other_data": "group2_eval2",
            },
            # Group 3: span_3 + attr_3 (unique, should keep)
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 45, 0),
                "args_span_id": {"span": "test_span_3"},
                "args_span_attribute": {"attr": "test_attr_3"},
                "score": 0.6,
                "other_data": "group3_unique",
            },
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should keep 4 rows: 2 from group1_eval3, 1 from group2_eval2, 1 from group3
        self.assertEqual(len(result), 4)
        expected_values = {
            "group1_eval3_row1",
            "group1_eval3_row2",
            "group2_eval2",
            "group3_unique",
        }
        self.assertEqual(set(result["other_data"].values), expected_values)

    def test_filter_duplicate_span_calls_with_string_values(self):
        """Test function works with string values in args_span_id and args_span_attribute."""
        test_data = pd.DataFrame([
            {
                "eval_root_id": "eval_1",
                "timestamp": datetime(2023, 1, 1, 10, 0, 0),
                "args_span_id": "string_span_1",
                "args_span_attribute": "string_attr_1",
                "score": 0.8,
                "other_data": "string_row_1",
            },
            {
                "eval_root_id": "eval_2",
                "timestamp": datetime(2023, 1, 1, 11, 0, 0),  # More recent
                "args_span_id": "string_span_1",
                "args_span_attribute": "string_attr_1",
                "score": 0.9,
                "other_data": "string_row_2",
            },
        ])

        result = _filter_duplicate_span_calls(test_data)

        # Should keep only the row from the most recent eval_root_id
        self.assertEqual(len(result), 1)
        self.assertEqual(result["eval_root_id"].iloc[0], "eval_2")
        self.assertEqual(result["other_data"].iloc[0], "string_row_2")


if __name__ == "__main__":
    unittest.main()
