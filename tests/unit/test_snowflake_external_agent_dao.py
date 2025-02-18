import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

try:
    from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
except Exception:
    pass


class DummyRow:
    def __init__(self, d: dict):
        self._d = d

    def as_dict(self):
        return self._d


@pytest.mark.snowflake
class TestExternalAgentDao(unittest.TestCase):
    def setUp(self):
        if ExternalAgentDao is None:
            self.skipTest(
                "ExternalAgentDao is not available because optional tests are disabled."
            )
        self.sf_session = MagicMock()

        # Dummy SQL execution: simulate that execute_query returns a list of rows,
        # which our DAO converts into a pandas DataFrame.
        dummy_sql = MagicMock()
        dummy_sql.collect.return_value = []
        self.sf_session.sql.return_value = dummy_sql
        self.dao = ExternalAgentDao(snowpark_session=self.sf_session)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_create_new_agent_with_version(self, mock_execute_query):
        # Simulate that the agent exists and that the version already exists.
        # Return a list of DummyRow objects to simulate rows.

        self.dao.create_new_agent("agent1", "v1")

        # Expect 2 calls (one for listing agents and one for listing versions).
        self.assertEqual(mock_execute_query.call_count, 1)

        # assert mock_execute_query is called with expected query
        expected_query = "CREATE EXTERNAL AGENT agent1 WITH VERSION v1;"
        mock_execute_query.assert_called_with(self.sf_session, expected_query)
