import unittest
from unittest.mock import MagicMock

import pandas as pd

try:
    from trulens.core.run import Run
except Exception:
    Run = None


def create_dummy_run() -> Run:
    base = {
        "run_name": "test_run",
        "object_name": "TEST_AGENT",
        "object_type": "EXTERNAL AGENT",
        "object_version": "v1",
        "run_metadata": Run.RunMetadata(),
        "source_info": {
            "name": "dummy_source",
            "column_spec": {"dummy": "dummy"},
            "source_type": "TABLE",
        },
    }
    extra = {
        "app": MagicMock(),
        "main_method_name": "dummy_method",
        "run_dao": MagicMock(),
        "tru_session": MagicMock(),
    }
    return Run.model_validate({**base, **extra})


class TestRunRecordAPIs(unittest.TestCase):
    def setUp(self):
        if Run is None:
            self.skipTest("TruLens Run class not available.")
            return

        self.run = create_dummy_run()

        # Mock the tru_session.get_records_and_feedback method
        self.mock_records_df = pd.DataFrame({
            "record_id": ["rec1", "rec2"],
            "input": ["input1", "input2"],
            "output": ["output1", "output2"],
            "latency": [100, 200],
            "metric1": [0.8, 0.9],
            "metric2": [0.7, 0.85],
            "extra_col": ["extra1", "extra2"],
        })
        self.mock_metrics_columns = ["metric1", "metric2"]

        self.run.tru_session.get_records_and_feedback.return_value = (
            self.mock_records_df,
            self.mock_metrics_columns,
        )

    def test_get_records_returns_overview_columns(self):
        result = self.run.get_records()

        expected_columns = [
            "record_id",
            "input",
            "output",
            "latency",
            "metric1",
            "metric2",
        ]
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 2)

    def test_get_records_with_record_ids(self):
        record_ids = ["rec1"]
        self.run.get_records(record_ids=record_ids)

        self.run.tru_session.get_records_and_feedback.assert_called_once_with(
            app_name="TEST_AGENT",
            app_version="v1",
            record_ids=record_ids,
            offset=None,
            limit=None,
        )

    def test_get_records_with_offset_limit(self):
        self.run.get_records(offset=10, limit=5)

        self.run.tru_session.get_records_and_feedback.assert_called_once_with(
            app_name="TEST_AGENT",
            app_version="v1",
            record_ids=None,
            offset=10,
            limit=5,
        )

    def test_get_record_details_returns_full_dataframe(self):
        result = self.run.get_record_details()

        pd.testing.assert_frame_equal(result, self.mock_records_df)

    def test_get_record_details_with_parameters(self):
        record_ids = ["rec1", "rec2"]
        self.run.get_record_details(record_ids=record_ids, offset=0, limit=10)

        self.run.tru_session.get_records_and_feedback.assert_called_once_with(
            app_name="TEST_AGENT",
            app_version="v1",
            record_ids=record_ids,
            offset=0,
            limit=10,
        )

    def test_get_records_uses_object_name_and_version(self):
        self.run.get_records()

        call_args = self.run.tru_session.get_records_and_feedback.call_args
        self.assertEqual(call_args[1]["app_name"], "TEST_AGENT")
        self.assertEqual(call_args[1]["app_version"], "v1")

    def test_get_record_details_uses_object_name_and_version(self):
        self.run.get_record_details()

        call_args = self.run.tru_session.get_records_and_feedback.call_args
        self.assertEqual(call_args[1]["app_name"], "TEST_AGENT")
        self.assertEqual(call_args[1]["app_version"], "v1")


if __name__ == "__main__":
    unittest.main()
