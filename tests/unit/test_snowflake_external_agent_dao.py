import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pandas as pd
import pytest
from trulens.core.schema import app as app_schema

try:
    from trulens.connectors.snowflake.dao.enums import ObjectType
    from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
    from trulens.connectors.snowflake.dao.sql_utils import escape_quotes


except Exception:
    pass

try:
    from trulens.connectors.snowflake.snowflake_event_table_db import (
        SnowflakeEventTableDB,
    )
except Exception:
    SnowflakeEventTableDB = None


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
    def test_create_agent_agent_not_exists(self, mock_execute_query):
        # Expect: first call (list_agents) returns empty; second call (create_new_agent) returns an empty DataFrame.
        mock_execute_query.side_effect = [
            pd.DataFrame(),
            pd.DataFrame(),
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
        # Return DataFrames to simulate results.
        mock_execute_query.side_effect = [
            pd.DataFrame([{"name": "AGENT1"}]),  # list_agents call
            pd.DataFrame([{"name": "v1"}]),  # list_agent_versions call
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
            pd.DataFrame([
                {"name": "AGENT 1"}
            ]),  # list_agents call returns agent exists
            pd.DataFrame(),  # list_agent_versions returns empty DataFrame
            pd.DataFrame(),  # add_version returns empty DataFrame
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
            pd.DataFrame([{"name": "AGENT1"}]),  # list_agents call
            pd.DataFrame([
                {"version": "v1"},
                {"version": "v2"},
            ]),  # list_agent_versions call
            pd.DataFrame(),  # add_version call returns empty DataFrame
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
            pd.DataFrame([{"name": "AGENT1"}]),  # list_agents call
        ]

        self.dao.drop_agent("agent1")

        expected_drop_query = 'DROP EXTERNAL AGENT "AGENT1";'
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        self.assertIn(expected_drop_query, queries)
        self.assertEqual(mock_execute_query.call_count, 1)

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_drop_default_version_raise(self, mock_execute_query):
        mock_execute_query.side_effect = [
            pd.DataFrame([
                {
                    "name": "v1",
                    "aliases": ["LAST", "DEFAULT", "FIRST"],
                }
            ]),  # list_agent_versions call
            pd.DataFrame(),
        ]

        with self.assertRaises(ValueError):
            # cannot drop default version
            self.dao.drop_current_version("agent1")

    @patch("trulens.connectors.snowflake.dao.external_agent.execute_query")
    def test_drop_current_version(self, mock_execute_query):
        mock_execute_query.side_effect = [
            pd.DataFrame([
                {"name": "v2", "aliases": ["LAST"]}
            ]),  # list_agent_versions call
            pd.DataFrame(),
        ]
        self.dao.drop_current_version("agent1")
        expected_query = (
            'ALTER EXTERNAL AGENT if exists "AGENT1" DROP VERSION "v2";'
        )
        calls = mock_execute_query.call_args_list
        queries = [call[0][1] for call in calls]
        self.assertIn(expected_query, queries)

        self.assertEqual(mock_execute_query.call_count, 2)

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


@pytest.mark.snowflake
class TestEventTableDBExternalApps(unittest.TestCase):
    """``SnowflakeEventTableDB.get_apps`` lists external agent versions."""

    def setUp(self):
        if SnowflakeEventTableDB is None:
            self.skipTest(
                "SnowflakeEventTableDB is not available because optional "
                "tests are disabled."
            )
        db = SnowflakeEventTableDB.__new__(SnowflakeEventTableDB)
        session = MagicMock()
        # SHOW AGENTS IN ACCOUNT reports no cortex agent, so the versions of
        # the named agent come from the external agent DAO.
        session.sql.return_value.to_pandas.return_value = pd.DataFrame({
            '"name"': []
        })
        db._snowpark_session = session
        db._external_agent_dao = MagicMock()
        db._external_agent_dao.list_agent_versions.return_value = pd.DataFrame({
            "name": ["v1", "v2"]
        })
        self.db = db

    def test_listed_apps_do_not_report_a_custom_app_id(self):
        """The app ids it builds are the computed ones, not custom values."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            apps = list(self.db.get_apps(app_name="my_agent"))

        expected = {
            (
                app_schema.AppDefinition._compute_app_id("my_agent", version),
                version,
            )
            for version in ("v1", "v2")
        }
        self.assertEqual(
            {(app["app_id"], app["app_version"]) for app in apps}, expected
        )
