import logging
import os

from snowflake.snowpark import Session
from trulens.apps.custom import TruCustomApp
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
            "TestSnowflakeEventTableExporter", append_uuid=True
        )
        db_connector = self._create_db_connector(self._snowpark_session)
        self._tru_session = TruSession(db_connector)

    def test_tru_app_unsupported_object_type(self):
        # Create app.
        app = TestApp()
        tru_recorder = TruCustomApp(
            app,
            app_name="custom app",
            app_version="v1",
            object_type="RANDOM_UNSUPPORTED",
        )

        self.assertIsNone(tru_recorder.snowflake_app_dao)

    def test_tru_app_supported_object_type(self):
        # Create app.
        app = TestApp()
        tru_recorder = TruCustomApp(
            app,
            app_name="custom app",
            app_version="v1",
            # object_type default to EXTERNAL_AGENT when snowflake connector is used
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_dao)

        agents_df = tru_recorder.snowflake_app_dao.list_agents()
        agent_names = agents_df["name"].values

        expected_fqn = f"{self._snowpark_session.get_current_database()}.{self._snowpark_session.get_current_schema()}.custom app"

        self.assertIn(expected_fqn, agent_names)

    def test_tru_app_multiple_versions(self):
        # Create app version 1.
        app_v1 = TestApp()
        tru_recorder_v1 = TruCustomApp(
            app_v1,
            app_name="custom app",
            app_version="v1",
        )

        self.assertIsNotNone(tru_recorder_v1.snowflake_app_dao)

        # Create app version 2.
        app_v2 = TestApp()
        tru_recorder_v2 = TruCustomApp(
            app_v2,
            app_name="custom app",
            app_version="v2",
        )

        self.assertIsNotNone(tru_recorder_v2.snowflake_app_dao)

        agents_df = tru_recorder_v1.snowflake_app_dao.list_agents()
        agent_names = agents_df["name"].values
        expected_fqn_v1 = f"{self._snowpark_session.get_current_database()}.{self._snowpark_session.get_current_schema()}.custom app.v1"
        expected_fqn_v2 = f"{self._snowpark_session.get_current_database()}.{self._snowpark_session.get_current_schema()}.custom app.v2"

        self.assertIn(expected_fqn_v1, agent_names)
        self.assertIn(expected_fqn_v2, agent_names)
