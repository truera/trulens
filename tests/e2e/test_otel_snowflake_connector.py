import pytest
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase
from tests.util.snowflake_test_case import SnowflakeTestCase


@pytest.mark.snowflake
class TestOtelSnowflakeConnector(OtelTestCase, SnowflakeTestCase):
    def test_connector_reconnects(self):
        self.create_and_use_schema(
            "test_connector_reconnects", append_uuid=True
        )
        connection_parameters = self._snowflake_connection_parameters.copy()
        connection_parameters["schema"] = self._schema
        connector = SnowflakeConnector(
            snowpark_session_creator=lambda: Session.builder.configs(
                connection_parameters
            ).create(),
            use_account_event_table=False,
        )
        tru_session = TruSession(connector=connector)
        self.assertEqual(connector, tru_session.connector)
        self.assertEqual(
            connector.snowpark_session, tru_session.connector.snowpark_session
        )
        self.assertNotEqual(connector.snowpark_session, self._snowpark_session)
        tru_session.get_apps()
        tru_session.connector.snowpark_session.close()
        try:
            tru_session.get_apps()
        except Exception:
            # Okay to fail here since the first time it may not realize the
            # connection was closed. The second time it should open a new one.
            pass
        tru_session.get_apps()
