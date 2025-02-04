import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from trulens.connectors.snowflake.dao.external_agent import ExternalAgentManager

from tests.test import optional_test


@optional_test
class TestExternalAgentManager(unittest.TestCase):
    def setUp(self):
        self.sf_session = MagicMock()

        self.sf_session.get_current_database.return_value = "DB"
        self.sf_session.get_current_schema.return_value = "SCH"

        # Dummy SQL execution: tests should override the responses
        dummy_sql = MagicMock()
        dummy_sql.collect.return_value = []
        self.sf_session.sql.return_value = dummy_sql

        self.manager = ExternalAgentManager(self.sf_session)

    @patch.object(ExternalAgentManager, "_execute_query")
    @patch.object(ExternalAgentManager, "_fetch_query")
    def test_create_agent_if_not_exist_agent_not_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        mock_fetch_query.return_value = []

        self.manager.create_agent_if_not_exist("agent1", "v1")

        expected_fqn = "DB.SCH.agent1"
        expected_query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        expected_parameters = (expected_fqn, "v1")
        expected_message = (
            f"Created External Agent {expected_fqn} with version v1."
        )

        mock_execute_query.assert_called_once_with(
            expected_query, expected_parameters, expected_message
        )

        mock_fetch_query.assert_called_once()

    @patch.object(ExternalAgentManager, "_execute_query")
    @patch.object(ExternalAgentManager, "_fetch_query")
    def test_create_agent_if_not_exist_agent_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        mock_fetch_query.return_value = ["DB.SCH.agent1"]

        self.manager.create_agent_if_not_exist("agent1", "v1")

        mock_execute_query.assert_not_called()

        mock_fetch_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
