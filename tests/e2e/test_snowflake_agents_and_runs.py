from datetime import datetime
import logging
import os

from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.run import RunConfig
from trulens.core.session import TruSession

from tests.unit.test_otel_tru_custom import TestApp
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeAgentsAndRuns(SnowflakeTestCase):
    logger = logging.getLogger(__name__)

    @staticmethod
    def _create_db_connector(snowpark_session: Session):
        return snowflake_connector.SnowflakeConnector(
            snowpark_session=snowpark_session,
        )

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["TRULENS_OTEL_TRACING"]
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.create_and_use_schema(
            "TestSnowflakeExternalAgent", append_uuid=True
        )
        db_connector = self._create_db_connector(self._snowpark_session)
        self.snowflake_connector = db_connector
        self._tru_session = TruSession(db_connector)

    def test_tru_app_unsupported_object_type(self):
        # Create app.
        app = TestApp()

        with self.assertRaises(ValueError):
            TruApp(
                app,
                app_name="custom_app",
                app_version="v1",
                connector=self.snowflake_connector,
                object_type="RANDOM_UNSUPPORTED",
            )

    def test_tru_app_missing_connector(self):
        # Create app.
        app = TestApp()

        tru_recorder = TruApp(
            app,
            app_name="custom_app",
            app_version="v1",
            main_method=app.respond_to_query,
        )

        self.assertIsNone(tru_recorder.snowflake_app_dao)

    def test_tru_app_snowflake_agent_initialization(self):
        # Create app.
        app = TestApp()
        TEST_APP_NAME = f"custom_app_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        tru_recorder = TruApp(
            app,
            app_name=TEST_APP_NAME,
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
            # object_type default to "EXTERNAL AGENT" when snowflake connector is used
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)
        self.assertIsNotNone(tru_recorder.snowflake_run_dao)

        self.assertEqual(tru_recorder.snowflake_object_type, "EXTERNAL AGENT")
        self.assertEqual(tru_recorder.snowflake_object_name, TEST_APP_NAME)

        self.assertTrue(
            tru_recorder.snowflake_app_dao.check_agent_exists(TEST_APP_NAME)
        )

        versions_df = tru_recorder.snowflake_app_dao.list_agent_versions(
            TEST_APP_NAME
        )

        self.assertIn(
            "v1", versions_df["name"].values
        )  # version is uppercased in snowflake

        tru_recorder.delete()

        self.assertFalse(
            tru_recorder.snowflake_app_dao.check_agent_exists(TEST_APP_NAME)
        )

    def test_tru_app_multiple_versions(self):
        # Create app version 1.

        TEST_APP_NAME = (
            f"custom_app_multi_ver_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        app = TestApp()
        tru_recorder_v1 = TruApp(
            app,
            app_name=TEST_APP_NAME,
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder_v1.snowflake_app_dao)
        self.assertTrue(
            tru_recorder_v1.snowflake_app_dao.check_agent_exists(TEST_APP_NAME)
        )
        # Create app version 2.
        tru_recorder_v2 = TruApp(
            app,
            app_name=TEST_APP_NAME,
            app_version="v2",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder_v2.snowflake_app_dao)

        self.assertTrue(
            tru_recorder_v2.snowflake_app_dao.check_agent_exists(TEST_APP_NAME)
        )

        versions_df_1 = tru_recorder_v1.snowflake_app_dao.list_agent_versions(
            TEST_APP_NAME
        )

        # # both versions should be present under the same agent, even created by 2 different truapp instances
        self.assertIn("v1", versions_df_1["name"].values)
        self.assertIn("v2", versions_df_1["name"].values)

        versions_df_2 = tru_recorder_v2.snowflake_app_dao.list_agent_versions(
            TEST_APP_NAME
        )
        self.assertIn("v1", versions_df_2["name"].values)
        self.assertIn("v2", versions_df_2["name"].values)

    def test_adding_run_missing_object_fields(self):
        app = TestApp()

        with self.assertRaises(Exception):
            TruApp(
                app,
                app_name=None,  #  missing object name, should raise error
                app_version="v1",
                connector=self.snowflake_connector,
                main_method=app.respond_to_query,
            )

    def test_adding_run_to_agent(self):
        app = TestApp()

        TEST_APP_NAME = f"custom_app_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        tru_recorder = TruApp(
            app,
            app_name=TEST_APP_NAME,
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)
        self.assertIsNotNone(tru_recorder.snowflake_run_dao)

        self.assertEqual(tru_recorder.snowflake_object_type, "EXTERNAL AGENT")
        self.assertEqual(tru_recorder.snowflake_object_name, TEST_APP_NAME)

        self.assertTrue(
            tru_recorder.snowflake_app_dao.check_agent_exists(TEST_APP_NAME)
        )

        versions_df = tru_recorder.snowflake_app_dao.list_agent_versions(
            TEST_APP_NAME
        )

        self.assertIn(
            "v1", versions_df["name"].values
        )  # version is uppercased in snowflake

        test_table_name = "test_table"
        self._snowpark_session.sql(
            f"create table if not exists {test_table_name} (name varchar)"
        ).collect()

        TEST_RUN_NAME = f"test_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        run_config = RunConfig(
            run_name=TEST_RUN_NAME,
            dataset_name=test_table_name,
            source_type="TABLE",
            label="label",
            dataset_spec={"input": "col1"},
            description="desc",
        )  # type: ignore

        new_run = tru_recorder.add_run(run_config=run_config)
        self.assertIsNotNone(new_run)

        run = tru_recorder.get_run(TEST_RUN_NAME)

        self.assertIsNotNone(run)
        self.assertEqual(new_run.run_name, TEST_RUN_NAME)

        self.assertEqual(new_run.description, "desc")

        self.assertDictEqual(run.model_dump(), new_run.model_dump())

        tru_recorder.delete()  # cleanup external agent after test

    def test_list_runs_after_adding(self):
        app = TestApp()

        TEST_APP_NAME = f"custom_app_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        tru_recorder = TruApp(
            app,
            app_name=TEST_APP_NAME,
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        test_table_name = "test_table"
        self._snowpark_session.sql(
            f"create table if not exists {test_table_name} (name varchar)"
        ).collect()

        TEST_RUN_NAME = f"test_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        invalid_dataset_spec_run_config = RunConfig(
            run_name=TEST_RUN_NAME,
            description="desc_1",
            dataset_name=test_table_name,
            dataset_spec={"gibberish": "col1"},
        )
        with self.assertRaises(ValueError):
            tru_recorder.add_run(run_config=invalid_dataset_spec_run_config)

        run_config_1 = RunConfig(
            run_name=TEST_RUN_NAME,
            description="desc_1",
            dataset_name=test_table_name,
            source_type="TABLE",
            dataset_spec={"input": "col1"},
        )
        tru_recorder.add_run(run_config=run_config_1)

        run_config_2 = RunConfig(
            run_name="test_run_2",
            description="desc_2",
            dataset_name=test_table_name,
            source_type="TABLE",
            dataset_spec={"input": "col1"},
        )
        run_2 = tru_recorder.add_run(run_config=run_config_2)

        run_2_ = tru_recorder.add_run(
            run_config=run_config_2
        )  # run already exists

        self.assertEqual(run_2.run_name, run_2_.run_name)

        runs = tru_recorder.list_runs()
        self.assertIn(TEST_RUN_NAME, [run.run_name for run in runs])
        self.assertIn("test_run_2", [run.run_name for run in runs])

        # test run deletion
        run_2.delete()
        runs = tru_recorder.list_runs()
        self.assertIn(TEST_RUN_NAME, [run.run_name for run in runs])
        self.assertNotIn("test_run_2", [run.run_name for run in runs])
