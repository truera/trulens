import json
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from trulens.connectors.snowflake.dao.run import RunDao


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

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_create_new_run(self, mock_execute_query):
        object_name = "MY_AGENT"
        object_type = "EXTERNAL AGENT"
        object_version = "V1"
        run_name = "my_run"
        dataset_name = "db.schema.table"
        source_type = "TABLE"
        dataset_spec = {"col1": "col1"}

        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "object_version": object_version,
            "run_name": run_name,
            "description": "desc",
            "run_metadata": {
                "labels": ["label"],
                "llm_judge_name": "mistral-large2",
            },
            "source_info": {
                "name": dataset_name,
                "column_spec": dataset_spec,
                "source_type": source_type,
            },
        }
        req_payload_json = json.dumps(req_payload)

        self.dao.create_new_run(
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
            dataset_name=dataset_name,
            source_type=source_type,
            dataset_spec=dataset_spec,
            description="desc",
            label="label",
            llm_judge_name="mistral-large2",
            run_name=run_name,
        )

        self.assertEqual(mock_execute_query.call_count, 2)

        for call in mock_execute_query.call_args_list:
            if call[0][1] == "SELECT SYSTEM$AIML_RUN_OPERATION('CREATE', ?);":
                actual_parameters = call[1].get("parameters", [])
                if actual_parameters:
                    actual_payload = json.loads(actual_parameters[0])
                    expected_payload = json.loads(req_payload_json)

                    print("Expected Payload:", expected_payload)
                    print("Actual Payload:", actual_payload)

                    # Perform deep comparison of the dictionaries (ignoring order)
                    self.assertEqual(actual_payload, expected_payload)

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_get_run_no_result(self, mock_execute_query):
        # Simulate that get_run returns an empty list (no run exists).
        mock_execute_query.return_value = []
        result_df = self.dao.get_run(
            run_name="nonexistent_run",
            object_name="MY_AGENT",
            object_type="EXTERNAL AGENT",
        )
        self.assertTrue(result_df.empty)

    @patch("trulens.connectors.snowflake.dao.run.execute_query")
    def test_get_run_with_result(self, mock_execute_query):
        # Simulate that get_run returns a single row.
        dummy = DummyRow({"run_name": "my_run", "run_status": "ACTIVE"})
        mock_execute_query.return_value = [dummy]
        result_df = self.dao.get_run(
            run_name="my_run",
            object_name="MY_AGENT",
            object_type="EXTERNAL AGENT",
        )
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.iloc[0]["run_name"], "my_run")
        self.assertEqual(result_df.iloc[0]["run_status"], "ACTIVE")

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
