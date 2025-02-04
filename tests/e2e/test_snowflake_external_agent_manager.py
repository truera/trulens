import logging
import os

from snowflake.snowpark import Session
from trulens.apps.custom import TruCustomApp
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.session import TruSession

from tests.unit.test_otel_tru_custom import TestApp
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeExternalAgentManager(SnowflakeTestCase):
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

        self.assertIsNone(tru_recorder.snowflake_app_manager)

    def test_tru_app_supported_object_type(self):
        # Create app.
        app = TestApp()
        tru_recorder = TruCustomApp(
            app,
            app_name="custom app",
            app_version="v1",
            object_type="EXTERNAL_AGENT",
        )

        self.assertIsNotNone(tru_recorder.snowflake_app_manager)

        # list_agents() to see if the agent has been created.
        agents = tru_recorder.snowflake_app_manager.list_agents()
        expected_fqn = f"{self._snowpark_session.get_current_database()}.{self._snowpark_session.get_current_schema()}.custom app"

        self.assertIn(expected_fqn, agents)
