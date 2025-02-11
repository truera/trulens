import json
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from trulens.connectors.snowflake.dao.run import RunDao
    from trulens.core.run import Run  # for Run.RunConfig
except Exception:
    RunDao = None


# DummyRow simulates a Snowflake Row with an as_dict() method.
class DummyRow:
    def __init__(self, d: dict):
        self._d = d

    def as_dict(self):
        return self._d


@pytest.mark.snowflake
class TestRunDao(unittest.TestCase):
    def setUp(self):
        if RunDao is None:
            self.skipTest(
                "RunDao is not available because optional tests are disabled."
            )

        self.sf_session = MagicMock()
        self.sf_session.get_current_database.return_value = "DB"
        self.sf_session.get_current_schema.return_value = "SCH"
        dummy_sql = MagicMock()
        dummy_sql.collect.return_value = []
        self.sf_session.sql.return_value = dummy_sql
        self.dao = RunDao(snowpark_session=self.sf_session)
        # Create a dummy RunConfig instance (fields used in create_new_run)
        self.run_config = Run.RunConfig(
            description="desc",
            label="label",
            dataset_fqn="db.schema.table",
            input_df=None,
            dataset_col_spec=None,
        )

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_create_new_run(self, mock_execute_query):
        object_name = "MY_AGENT"
        object_type = "EXTERNAL AGENT"
        run_name = "my_run"

        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "description": self.run_config.description,
            "label": self.run_config.label,
        }
        req_payload_json = json.dumps(req_payload)
        expected_query = "SELECT SYSTEM$AIML_RUN_OPERATION('CREATE', ?);"

        self.dao.create_new_run(
            object_name, object_type, run_name, self.run_config
        )
        mock_execute_query.assert_any_call(
            self.sf_session,
            expected_query,
            parameters=(req_payload_json,),
        )
        self.assertEqual(mock_execute_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_get_run_no_result(self, mock_execute_query):
        # Simulate that get_run returns an empty list (no run exists).
        mock_execute_query.return_value = []
        result_df = self.dao.get_run("MY_AGENT", "nonexistent_run")
        self.assertTrue(result_df.empty)

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_get_run_with_result(self, mock_execute_query):
        # Simulate that get_run returns a single row.
        dummy = DummyRow({"run_name": "my_run", "run_status": "ACTIVE"})
        mock_execute_query.return_value = [dummy]
        result_df = self.dao.get_run("MY_AGENT", "my_run")
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.iloc[0]["run_name"], "my_run")
        self.assertEqual(result_df.iloc[0]["run_status"], "ACTIVE")

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_list_all_runs(self, mock_execute_query):
        # Simulate list_all_runs returning multiple rows.
        dummy1 = DummyRow({"run_name": "run1", "run_status": "ACTIVE"})
        dummy2 = DummyRow({"run_name": "run2", "run_status": "INACTIVE"})
        mock_execute_query.return_value = [dummy1, dummy2]
        result_df = self.dao.list_all_runs("MY_AGENT", "EXTERNAL AGENT")
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 1)
        self.assertIn("run1", result_df["run_name"].values)
        self.assertIn("run2", result_df["run_name"].values)

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_delete_run(self, mock_execute_query):
        req_payload = {
            "run_name": "my_run",
            "object_name": "MY_AGENT",
            "object_type": "EXTERNAL AGENT",
        }
        req_payload_json = json.dumps(req_payload)
        expected_query = "SELECT SYSTEM$AIML_RUN_OPERATION('DELETE', ?);"
        self.dao.delete_run("my_run", "MY_AGENT", "EXTERNAL AGENT")
        mock_execute_query.assert_called_once_with(
            self.sf_session, expected_query, parameters=(req_payload_json,)
        )
