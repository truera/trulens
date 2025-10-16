import unittest
from unittest.mock import MagicMock

import pandas as pd
import pytest
from snowflake.snowpark.row import Row

try:
    from trulens.connectors.snowflake.dao.sql_utils import execute_query
except ImportError:
    pass


@pytest.mark.snowflake
class TestExecuteQuery(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.mock_sql_result = MagicMock()
        self.mock_session.sql.return_value = self.mock_sql_result

    def test_select_query_success(self):
        expected_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        self.mock_sql_result.to_pandas.return_value = expected_df

        result = execute_query(self.mock_session, "SELECT * FROM table")

        self.mock_session.sql.assert_called_once_with(
            "SELECT * FROM table", params=None
        )
        self.mock_sql_result.to_pandas.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)

    def test_select_query_with_parameters(self):
        expected_df = pd.DataFrame({"id": [1], "name": ["test"]})
        self.mock_sql_result.to_pandas.return_value = expected_df
        params = (1, "test")

        result = execute_query(
            self.mock_session,
            "SELECT * FROM table WHERE id = ? AND name = ?",
            params,
        )

        self.mock_session.sql.assert_called_once_with(
            "SELECT * FROM table WHERE id = ? AND name = ?", params=params
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_non_select_query_with_results(self):
        mock_row1 = MagicMock(spec=Row)
        mock_row1._fields = ["count"]
        mock_row1.__iter__ = lambda self: iter([5])

        mock_row2 = MagicMock(spec=Row)
        mock_row2._fields = ["count"]
        mock_row2.__iter__ = lambda self: iter([10])

        self.mock_sql_result.collect.return_value = [mock_row1, mock_row2]

        result = execute_query(
            self.mock_session, "INSERT INTO table VALUES (1)"
        )

        self.mock_session.sql.assert_called_once_with(
            "INSERT INTO table VALUES (1)", params=None
        )
        self.mock_sql_result.collect.assert_called_once()
        expected_df = pd.DataFrame({"count": [5, 10]})
        pd.testing.assert_frame_equal(result, expected_df)

    def test_non_select_query_empty_result(self):
        self.mock_sql_result.collect.return_value = []

        result = execute_query(
            self.mock_session, "DELETE FROM table WHERE id = 999"
        )

        self.mock_session.sql.assert_called_once_with(
            "DELETE FROM table WHERE id = 999", params=None
        )
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)

    def test_column_name_cleanup(self):
        df_with_quoted_cols = pd.DataFrame({
            '"QUOTED_COL"': [1],
            "NORMAL_COL": [2],
        })
        self.mock_sql_result.to_pandas.return_value = df_with_quoted_cols

        result = execute_query(self.mock_session, "SELECT * FROM table")

        expected_columns = ["QUOTED_COL", "NORMAL_COL"]
        self.assertEqual(list(result.columns), expected_columns)

    def test_query_execution_error(self):
        self.mock_session.sql.side_effect = Exception(
            "Database connection error"
        )

        with self.assertRaises(Exception) as context:
            execute_query(self.mock_session, "SELECT * FROM table")

        self.assertEqual(str(context.exception), "Database connection error")

    def test_case_insensitive_select_detection(self):
        expected_df = pd.DataFrame({"result": [1]})
        self.mock_sql_result.to_pandas.return_value = expected_df

        result = execute_query(self.mock_session, "  select * from table  ")

        self.mock_sql_result.to_pandas.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)

    def test_mixed_case_select_detection(self):
        expected_df = pd.DataFrame({"result": [1]})
        self.mock_sql_result.to_pandas.return_value = expected_df

        result = execute_query(self.mock_session, "Select * FROM table")

        self.mock_sql_result.to_pandas.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)
