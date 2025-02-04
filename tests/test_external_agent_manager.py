import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from tests.test import optional_test
from tests.test import run_optional_tests

if run_optional_tests():
    from trulens.connectors.snowflake.dao.external_agent import (
        ExternalAgentManager,
    )


@optional_test
class TestExternalAgentManager(unittest.TestCase):
    def setUp(self):
        self.sf_session = MagicMock()
        # Configure the dummy session to return fixed database and schema names.
        self.sf_session.get_current_database.return_value = "DB"
        self.sf_session.get_current_schema.return_value = "SCH"

        # Dummy SQL execution: we'll let our tests override the responses
        dummy_sql = MagicMock()
        dummy_sql.collect.return_value = []
        self.sf_session.sql.return_value = dummy_sql

        # Create an instance of ExternalAgentManager with the dummy session.
        self.manager = ExternalAgentManager(self.sf_session)

    @patch.object(ExternalAgentManager, "_execute_query")
    @patch.object(ExternalAgentManager, "_fetch_query")
    def test_create_agent_if_not_exist_agent_not_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        # Simulate that no agents currently exist.
        mock_fetch_query.return_value = []

        # Call create_agent_if_not_exist. Since the agent is not present, it should create it.
        self.manager.create_agent_if_not_exist("agent1", "v1")

        # The fully qualified name should be composed using database and schema.
        expected_fqn = "DB.SCH.agent1"
        expected_query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        expected_parameters = (expected_fqn, "v1")
        expected_message = (
            f"Created External Agent {expected_fqn} with version v1."
        )

        # Assert that _execute_query was called once with the expected query and parameters.
        mock_execute_query.assert_called_once_with(
            expected_query, expected_parameters, expected_message
        )
        # And _fetch_query should have been called to list the agents.
        mock_fetch_query.assert_called_once()

    @patch.object(ExternalAgentManager, "_execute_query")
    @patch.object(ExternalAgentManager, "_fetch_query")
    def test_create_agent_if_not_exist_agent_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        # Simulate that the agent already exists.
        mock_fetch_query.return_value = ["DB.SCH.agent1"]

        # Call create_agent_if_not_exist. Since the agent exists, it should log an info message and do nothing.
        self.manager.create_agent_if_not_exist("agent1", "v1")

        # In this case, _execute_query should not be called.
        mock_execute_query.assert_not_called()
        # And _fetch_query should have been called once.
        mock_fetch_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
