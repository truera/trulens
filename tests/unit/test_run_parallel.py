import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
from trulens.core.dao.run import RunDaoBase
from trulens.core.run import Run
from trulens.core.run import RunConfig


def _make_run(**overrides):
    mock_app = MagicMock()
    mock_app.app_name = "test"
    mock_app.app_version = "1.0"

    defaults = dict(
        run_dao=MagicMock(spec=RunDaoBase),
        app=mock_app,
        main_method_name="invoke",
        tru_session=MagicMock(),
        object_name="TEST",
        object_type="EXTERNAL AGENT",
        object_version="1.0",
        run_name="test_run",
        run_metadata=Run.RunMetadata(),
        source_info=Run.SourceInfo(
            name="test", column_spec={}, source_type="TABLE"
        ),
    )
    defaults.update(overrides)
    return Run(**defaults)


class TestRunConfigWorkerKnobs(unittest.TestCase):
    def test_defaults_are_none(self):
        rc = RunConfig(
            run_name="r",
            dataset_name="d",
            dataset_spec={"input": "col"},
        )
        self.assertIsNone(rc.invocation_max_workers)
        self.assertIsNone(rc.metric_max_workers)

    def test_explicit_values(self):
        rc = RunConfig(
            run_name="r",
            dataset_name="d",
            dataset_spec={"input": "col"},
            invocation_max_workers=8,
            metric_max_workers=2,
        )
        self.assertEqual(rc.invocation_max_workers, 8)
        self.assertEqual(rc.metric_max_workers, 2)


class TestRunWorkerKnobs(unittest.TestCase):
    def test_worker_fields_excluded_from_serialization(self):
        run = _make_run(invocation_max_workers=4, metric_max_workers=2)
        dumped = run.model_dump()
        self.assertNotIn("invocation_max_workers", dumped)
        self.assertNotIn("metric_max_workers", dumped)

    def test_worker_fields_stored_on_instance(self):
        run = _make_run(invocation_max_workers=4, metric_max_workers=2)
        self.assertEqual(run.invocation_max_workers, 4)
        self.assertEqual(run.metric_max_workers, 2)


class TestSnowflakeWarning(unittest.TestCase):
    @patch("trulens.core.run.logger")
    def test_warning_emitted_for_snowflake_connector(self, mock_logger):
        mock_session = MagicMock()
        mock_connector = MagicMock()
        type(mock_connector).__name__ = "SnowflakeConnector"
        mock_session.connector = mock_connector

        run = _make_run(tru_session=mock_session, metric_max_workers=3)
        run._warn_if_snowflake_parallel(3)
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "metric_max_workers=3", mock_logger.warning.call_args[0][0]
        )

    @patch("trulens.core.run.logger")
    def test_no_warning_for_non_snowflake(self, mock_logger):
        mock_session = MagicMock()
        mock_connector = MagicMock()
        type(mock_connector).__name__ = "DefaultConnector"
        mock_session.connector = mock_connector

        run = _make_run(tru_session=mock_session, metric_max_workers=3)
        run._warn_if_snowflake_parallel(3)
        mock_logger.warning.assert_not_called()

    @patch("trulens.core.run.logger")
    def test_no_warning_when_single_worker(self, mock_logger):
        mock_session = MagicMock()
        mock_connector = MagicMock()
        type(mock_connector).__name__ = "SnowflakeConnector"
        mock_session.connector = mock_connector

        run = _make_run(tru_session=mock_session, metric_max_workers=1)
        run._warn_if_snowflake_parallel(1)
        mock_logger.warning.assert_not_called()


class TestInvokeSingleRow(unittest.TestCase):
    def test_invoke_single_row_calls_instrumented_method(self):
        run = _make_run()
        row = pd.Series({"question": "hello"})
        dataset_spec = {"input": "question"}

        run._invoke_single_row(row, dataset_spec, input_records_count=1)

        run.app.instrumented_invoke_main_method.assert_called_once()
        call_kwargs = run.app.instrumented_invoke_main_method.call_args[1]
        self.assertEqual(call_kwargs["run_name"], "test_run")
        self.assertEqual(call_kwargs["input_records_count"], 1)
        self.assertIn("hello", call_kwargs["main_method_args"])

    def test_invoke_single_row_with_ground_truth(self):
        run = _make_run()
        row = pd.Series({"question": "hi", "gt": "expected"})
        dataset_spec = {"input": "question", "ground_truth_output": "gt"}

        run._invoke_single_row(row, dataset_spec, input_records_count=1)

        call_kwargs = run.app.instrumented_invoke_main_method.call_args[1]
        self.assertEqual(call_kwargs["ground_truth_output"], "expected")


class TestStartParallelism(unittest.TestCase):
    @patch("trulens.core.run.as_completed", side_effect=lambda fs: iter(fs))
    @patch("trulens.core.run.ThreadPoolExecutor")
    @patch.object(Run, "_can_start_new_invocation", return_value=True)
    @patch.object(Run, "get_status", return_value="CREATED")
    def test_start_uses_invocation_max_workers(
        self, mock_status, mock_can_start, mock_pool_cls, mock_as_completed
    ):
        run = _make_run(invocation_max_workers=2)

        mock_pool = MagicMock()
        mock_pool_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_pool.submit.return_value = mock_future

        input_df = pd.DataFrame({"question": ["a", "b", "c"]})
        run.source_info.column_spec = {"input": "question"}

        run.start(input_df)

        mock_pool_cls.assert_called_once_with(max_workers=2)

    @patch("trulens.core.run.as_completed", side_effect=lambda fs: iter(fs))
    @patch("trulens.core.run.ThreadPoolExecutor")
    @patch.object(Run, "_can_start_new_invocation", return_value=True)
    @patch.object(Run, "get_status", return_value="CREATED")
    def test_start_default_workers_capped_at_4(
        self, mock_status, mock_can_start, mock_pool_cls, mock_as_completed
    ):
        run = _make_run()

        mock_pool = MagicMock()
        mock_pool_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_pool.submit.return_value = mock_future

        input_df = pd.DataFrame({"question": [f"q{i}" for i in range(20)]})
        run.source_info.column_spec = {"input": "question"}

        run.start(input_df)

        mock_pool_cls.assert_called_once_with(max_workers=4)


if __name__ == "__main__":
    unittest.main()
