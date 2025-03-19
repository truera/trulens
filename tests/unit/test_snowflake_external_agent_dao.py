import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from trulens.connectors.snowflake.dao.enums import ObjectType
    from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
    from trulens.connectors.snowflake.dao.sql_utils import escape_quotes


except Exception:
    pass


class DummyRow:
    def __init__(self, d: dict):
        self._d = d

    def as_dict(self):
        return self._d


# @pytest.mark.snowflake
@pytest.mark.optional
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
    def test_create_agent_agent_not_exists(self, mock_execute_query):
        # Expect: first call (list_agents) returns empty; second call (create_new_agent) returns an empty DataFrame.
        mock_execute_query.side_effect = [
            [],
            [],
        ]  # first call: list_agents, second: create_new_agent

        self.dao.create_agent_if_not_exist("agent1", "v1")

        expected_show_query = "SHOW EXTERNAL AGENTS;"
        expected_create_query = (
            'CREATE EXTERNAL AGENT "AGENT1" WITH VERSION "v1";'
        )

        # We expect exactly 2 calls to execute_query.
        self.assertEqual(mock_execute_query.call_count, 2)

        # Extract the list of calls:
        calls = mock_execute_query.call_args_list

        # Check that one of the calls has the expected SHOW query and one has the expected CREATE query.
        queries = [
            call[0][1] for call in calls
        ]  # call[0][1] extracts the 'query' argument from each call.

        self.assertIn(expected_show_query, queries)
        self.assertIn(expected_create_query, queries)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_create_agent_agent_exists_version_exists(self, mock_execute_query):
        # Simulate that the agent exists and that the version already exists.
        # Return a list of DummyRow objects to simulate rows.
        mock_execute_query.side_effect = [
            [DummyRow({"name": "AGENT1"})],  # list_agents call
            [DummyRow({"name": "v1"})],  # list_agent_versions call
        ]

        self.dao.create_agent_if_not_exist("agent1", "v1")

        # Expect 2 calls (one for listing agents and one for listing versions).
        self.assertEqual(mock_execute_query.call_count, 2)
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        expected_show_agents_query = "SHOW EXTERNAL AGENTS;"
        expected_show_versions_query = (
            'SHOW VERSIONS IN EXTERNAL AGENT "AGENT1";'
        )
        self.assertIn(expected_show_agents_query, queries)
        self.assertIn(expected_show_versions_query, queries)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_create_agent_agent_exists_version_not_exists_empty_list(
        self, mock_execute_query
    ):
        # Simulate that the agent exists.
        mock_execute_query.side_effect = [
            [
                DummyRow({"name": "AGENT 1"})
            ],  # list_agents call returns agent exists
            [],  # list_agent_versions returns empty list
            [],  # add_version returns empty list
        ]

        self.dao.create_agent_if_not_exist("agent 1", "v2")

        expected_add_query = (
            'ALTER EXTERNAL AGENT if exists "AGENT 1"  ADD VERSION "v2";'
        )
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        self.assertIn(expected_add_query, queries)
        self.assertEqual(mock_execute_query.call_count, 3)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_create_agent_agent_exists_version_not_in_existing_list(
        self, mock_execute_query
    ):
        # Simulate that the agent exists and its versions are present but do not include "V3"
        mock_execute_query.side_effect = [
            [DummyRow({"name": "AGENT1"})],  # list_agents call
            [
                DummyRow({"version": "v1"}),
                DummyRow({"version": "v2"}),
            ],  # list_agent_versions call
            [],  # add_version call returns empty list
        ]

        self.dao.create_agent_if_not_exist("agent1", "v3")

        expected_add_query = (
            'ALTER EXTERNAL AGENT if exists "AGENT1"  ADD VERSION "v3";'
        )
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        self.assertIn(expected_add_query, queries)
        self.assertEqual(mock_execute_query.call_count, 3)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_delete_agent(self, mock_execute_query):
        # Simulate that the agent exists.
        mock_execute_query.side_effect = [
            [DummyRow({"name": "AGENT1"})],  # list_agents call
        ]

        self.dao.drop_agent("agent1")

        expected_drop_query = 'DROP EXTERNAL AGENT "AGENT1";'
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        self.assertIn(expected_drop_query, queries)
        self.assertEqual(mock_execute_query.call_count, 1)

    @patch.object(ExternalAgentDao, "list_agent_versions")
    def test_get_current_version_success(self, mock_list_agent_versions):
        # Simulate list_agent_versions returning a DataFrame with two rows,
        # where one row's "aliases" contains "LAST".
        data = [
            {"aliases": ["FIRST", "DEFAULT"], "name": "v0"},
            {"aliases": ["LAST", "DEFAULT"], "name": "v1"},
        ]
        df = pd.DataFrame(data)
        mock_list_agent_versions.return_value = df

        current_version = self.dao._get_current_version("agent1")
        self.assertEqual(current_version, "v1")

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    @patch.object(ExternalAgentDao, "_get_current_version", return_value="v1")
    def test_drop_current_version(
        self, mock_get_current_version, mock_execute_query
    ):
        # When drop_current_version is called, _get_current_version returns "v1".
        # The agent name should be resolved to upper-case.
        self.dao.drop_current_version("agent1")
        expected_query = (
            'ALTER EXTERNAL AGENT if exists "AGENT1" DROP VERSION "v1";'
        )
        mock_execute_query.assert_called_once_with(
            self.sf_session, expected_query
        )

    def test_is_valid_object(self):
        self.assertTrue(ObjectType.EXTERNAL_AGENT == "EXTERNAL AGENT")
        self.assertTrue(ObjectType.is_valid_object("EXTERNAL AGENT"))
        self.assertFalse(ObjectType.is_valid_object("INVALID AGENT"))

    def test_escape_quotes(self):
        self.assertEqual(escape_quotes('hello "world"'), 'hello ""world""')
        self.assertEqual(
            escape_quotes('he said "hello" and "goodbye"'),
            'he said ""hello"" and ""goodbye""',
        )
        self.assertEqual(escape_quotes('""""'), '""""""""')
        self.assertEqual(escape_quotes(""), "")
