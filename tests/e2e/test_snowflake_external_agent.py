import logging
import os

from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.run import Run
from trulens.core.session import TruSession

from tests.unit.test_otel_tru_custom import TestApp
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeExternalAgentDao(SnowflakeTestCase):
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

        tru_recorder = TruApp(
            app,
            app_name="custom_app",
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
            # object_type default to "EXTERNAL AGENT" when snowflake connector is used
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)
        self.assertIsNotNone(tru_recorder.snowflake_run_dao)

        self.assertEqual(tru_recorder.snowflake_object_type, "EXTERNAL AGENT")
        self.assertEqual(tru_recorder.snowflake_object_name, "CUSTOM_APP")

        self.assertTrue(
            tru_recorder.snowflake_app_dao.check_agent_exists("custom_app")
        )

        versions_df = tru_recorder.snowflake_app_dao.list_agent_versions(
            "custom_app"
        )

        self.assertIn(
            "V1", versions_df["name"].values
        )  # version is uppercased in snowflake

        tru_recorder.delete_snowflake_app()

        self.assertTrue(tru_recorder.snowflake_app_dao.list_agents().empty)

    def test_tru_app_multiple_versions(self):
        # Create app version 1.
        app = TestApp()
        tru_recorder_v1 = TruApp(
            app,
            app_name="custom_app_multi_ver",
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder_v1.snowflake_app_dao)
        self.assertTrue(
            tru_recorder_v1.snowflake_app_dao.check_agent_exists(
                "custom_app_multi_ver"
            )
        )
        # Create app version 2.
        tru_recorder_v2 = TruApp(
            app,
            app_name="custom_app_multi_ver",
            app_version="v2",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder_v2.snowflake_app_dao)

        self.assertTrue(
            tru_recorder_v2.snowflake_app_dao.check_agent_exists(
                "custom_app_multi_ver"
            )
        )

        versions_df_1 = tru_recorder_v1.snowflake_app_dao.list_agent_versions(
            "custom_app_multi_ver"
        )

        # # both versions should be present under the same agent, even created by 2 different truapp instances
        self.assertIn("V1", versions_df_1["name"].values)
        self.assertIn("V2", versions_df_1["name"].values)

        versions_df_2 = tru_recorder_v2.snowflake_app_dao.list_agent_versions(
            "custom_app_multi_ver"
        )
        self.assertIn("V1", versions_df_2["name"].values)
        self.assertIn("V2", versions_df_2["name"].values)

    def test_adding_run_to_agent(self):
        app = TestApp()

        tru_recorder = TruApp(
            app,
            app_name="custom_app",
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)
        self.assertIsNotNone(tru_recorder.snowflake_run_dao)

        self.assertEqual(tru_recorder.snowflake_object_type, "EXTERNAL AGENT")
        self.assertEqual(tru_recorder.snowflake_object_name, "CUSTOM_APP")

        self.assertTrue(
            tru_recorder.snowflake_app_dao.check_agent_exists("custom_app")
        )

        versions_df = tru_recorder.snowflake_app_dao.list_agent_versions(
            "custom_app"
        )

        self.assertIn(
            "V1", versions_df["name"].values
        )  # version is uppercased in snowflake

        run_config = Run.RunConfig(
            description="desc",
            dataset_fqn="db.schema.table",
            dataset_col_spec=None,
        )  # type: ignore
        new_run = tru_recorder.add_run(
            run_name="test_run_1", run_config=run_config
        )
        self.assertIsNotNone(new_run)

        run = tru_recorder.get_run("test_run_1")

        self.assertIsNotNone(run)
        self.assertEqual(new_run.run_name, "test_run_1")

        self.assertEqual(new_run.description, "desc")

        self.assertDictEqual(run.model_dump(), new_run.model_dump())

    def test_list_runs_after_adding(self):
        app = TestApp()

        tru_recorder = TruApp(
            app,
            app_name="custom_app",
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
        )

        run_config_1 = Run.RunConfig(
            description="desc_1",
            dataset_fqn="db.schema.table",
            dataset_col_spec=None,
        )
        tru_recorder.add_run(run_name="test_run_1", run_config=run_config_1)

        run_config_2 = Run.RunConfig(
            description="desc_2",
            dataset_fqn="db.schema.table",
            dataset_col_spec=None,
        )
        run_2 = tru_recorder.add_run(
            run_name="test_run_2", run_config=run_config_2
        )

        runs = tru_recorder.list_runs()
        self.assertIn("test_run_1", [run.run_name for run in runs])
        self.assertIn("test_run_2", [run.run_name for run in runs])

        # test run deletion
        run_2.delete()
        runs = tru_recorder.list_runs()
        self.assertIn("test_run_1", [run.run_name for run in runs])
        self.assertNotIn("test_run_2", [run.run_name for run in runs])
