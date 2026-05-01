import json
import unittest

from trulens.core.dao.default_run import DefaultRunDao
from trulens.core.dao.run import RunDaoBase
from trulens.core.database.orm import make_orm_for_prefix
from trulens.core.enums import Mode
from trulens.core.run import Run


class FakeDB:
    """Minimal DB stand-in that provides an ORM and sessionmaker for testing."""

    _counter = 0

    def __init__(self):
        import sqlalchemy as sa
        from sqlalchemy.orm import sessionmaker

        FakeDB._counter += 1
        self.engine = sa.create_engine("sqlite:///:memory:")
        self.orm = make_orm_for_prefix(
            table_prefix=f"test_dao_{FakeDB._counter}_"
        )
        self.orm.metadata.create_all(self.engine)
        self.session = sessionmaker(self.engine)


class TestDefaultRunDao(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        self.dao = DefaultRunDao(db=self.db)
        self.object_name = "test_app"
        self.object_type = "APP"
        self.object_version = "v1"

    def test_isinstance_of_base(self):
        self.assertIsInstance(self.dao, RunDaoBase)

    def test_create_and_get_run(self):
        df = self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
            run_name="run_1",
            dataset_name="my_dataset",
            source_type="DATAFRAME",
            dataset_spec={"input": "question_col"},
            description="test run",
            label="test_label",
        )
        self.assertFalse(df.empty)
        metadata = json.loads(df.iloc[0].values[0])
        self.assertEqual(metadata["run_name"], "run_1")
        self.assertEqual(metadata["object_name"], self.object_name)
        self.assertEqual(metadata["source_info"]["name"], "my_dataset")
        self.assertEqual(metadata["run_metadata"]["labels"], ["test_label"])

        fetched = self.dao.get_run(
            run_name="run_1",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        self.assertFalse(fetched.empty)
        fetched_meta = json.loads(fetched.iloc[0].values[0])
        self.assertEqual(fetched_meta["run_name"], "run_1")

    def test_create_duplicate_raises(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="dup_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )
        with self.assertRaises(ValueError):
            self.dao.create_new_run(
                object_name=self.object_name,
                object_type=self.object_type,
                run_name="dup_run",
                dataset_name="ds",
                source_type="DATAFRAME",
                dataset_spec={},
            )

    def test_list_all_runs(self):
        for i in range(3):
            self.dao.create_new_run(
                object_name=self.object_name,
                object_type=self.object_type,
                run_name=f"run_{i}",
                dataset_name="ds",
                source_type="DATAFRAME",
                dataset_spec={},
            )
        df = self.dao.list_all_runs(
            object_name=self.object_name,
            object_type=self.object_type,
        )
        self.assertFalse(df.empty)
        runs = json.loads(df.iloc[0].iloc[-1])
        self.assertEqual(len(runs), 3)

    def test_list_runs_empty(self):
        df = self.dao.list_all_runs(
            object_name="nonexistent",
            object_type=self.object_type,
        )
        self.assertTrue(df.empty)

    def test_delete_run(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="to_delete",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )
        self.dao.delete_run(
            run_name="to_delete",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        df = self.dao.get_run(
            run_name="to_delete",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        self.assertTrue(df.empty)

    def test_upsert_metadata_fields(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="meta_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )

        self.dao.upsert_run_metadata_fields(
            run_name="meta_run",
            object_name=self.object_name,
            object_type=self.object_type,
            run_status="CANCELLED",
        )

        df = self.dao.get_run(
            run_name="meta_run",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        meta = json.loads(df.iloc[0].values[0])
        self.assertEqual(meta["run_status"], "CANCELLED")

    def test_upsert_invocation_metadata(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="inv_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )

        self.dao.upsert_run_metadata_fields(
            run_name="inv_run",
            object_name=self.object_name,
            object_type=self.object_type,
            entry_id="inv_1",
            entry_type="invocations",
            id="inv_1",
            input_records_count=10,
            start_time_ms=1000,
        )

        df = self.dao.get_run(
            run_name="inv_run",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        meta = json.loads(df.iloc[0].values[0])
        invocations = meta["run_metadata"]["invocations"]
        self.assertIn("inv_1", invocations)
        self.assertEqual(invocations["inv_1"]["input_records_count"], 10)

    def test_start_ingestion_query(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="ingest_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )

        self.dao.start_ingestion_query(
            object_name=self.object_name,
            object_version=self.object_version,
            object_type=self.object_type,
            run_name="ingest_run",
            input_records_count=5,
        )

        df = self.dao.get_run(
            run_name="ingest_run",
            object_name=self.object_name,
            object_type=self.object_type,
        )
        meta = json.loads(df.iloc[0].values[0])
        invocations = meta["run_metadata"]["invocations"]
        self.assertTrue(len(invocations) > 0)
        inv = list(invocations.values())[0]
        self.assertEqual(inv["input_records_count"], 5)
        self.assertEqual(inv["completion_status"]["status"], "COMPLETED")
        self.assertEqual(inv["completion_status"]["record_count"], 5)

    def test_call_compute_metrics_query_warns(self):
        self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="metrics_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={},
        )

        from unittest.mock import patch

        with patch("trulens.core.dao.default_run.logger") as mock_logger:
            self.dao.call_compute_metrics_query(
                metrics=["accuracy"],
                object_name=self.object_name,
                object_version=self.object_version,
                object_type=self.object_type,
                run_name="metrics_run",
            )
            mock_logger.warning.assert_called_once()

    def test_from_metadata_df_round_trip(self):
        df = self.dao.create_new_run(
            object_name=self.object_name,
            object_type=self.object_type,
            run_name="rt_run",
            dataset_name="ds",
            source_type="DATAFRAME",
            dataset_spec={"input": "q"},
            description="round trip test",
            mode=Mode.APP_INVOCATION,
        )

        from unittest.mock import MagicMock

        mock_app = MagicMock()
        mock_session = MagicMock()

        run = Run.from_metadata_df(
            df,
            {
                "app": mock_app,
                "main_method_name": "invoke",
                "run_dao": self.dao,
                "tru_session": mock_session,
            },
        )
        self.assertEqual(run.run_name, "rt_run")
        self.assertEqual(run.object_name, self.object_name)
        self.assertEqual(run.source_info.name, "ds")
        self.assertEqual(run.source_info.column_spec, {"input": "q"})


if __name__ == "__main__":
    unittest.main()
