import logging
import os

from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.connectors import snowflake as snowflake_connector
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

    def test_tru_app_otel_enabled_missing_main_method(self):
        # Create app.
        app = TestApp()

        with self.assertRaises(ValueError):
            TruApp(
                app,
                app_name="custom_app",
                app_version="v1",
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

    def test_tru_app_supported_object_type(self):
        # Create app.
        app = TestApp()

        tru_recorder = TruApp(
            app,
            app_name="custom_app",
            app_version="v1",
            connector=self.snowflake_connector,
            main_method=app.respond_to_query,
            # object_type default to EXTERNAL_AGENT when snowflake connector is used
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)

        self.assertTrue(
            tru_recorder.snowflake_app_dao.check_agent_exists("custom_app")
        )

        versions_df = tru_recorder.snowflake_app_dao.list_agent_versions(
            "custom_app"
        )

        self.assertIn(
            "V1", versions_df["name"].values
        )  # version is uppercased in snowflake

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
