import sys
from unittest import main
from unittest import skipIf
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from tests.test import TruTestCase

try:
    from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
except Exception:
    pass


@skipIf(
    sys.version_info >= (3, 12),
    "trulens-connector-snowflake is not yet supported in Python 3.12",
)
@pytest.mark.optional
class TestExternalAgentDao(TruTestCase):
    def setUp(self):
        if ExternalAgentDao is None:
            self.skipTest(
                "ExternalAgentDao is not available because optional tests are disabled."
            )
        self.sf_session = MagicMock()
        self.sf_session.get_current_database.return_value = "DB"
        self.sf_session.get_current_schema.return_value = "SCH"
        self.dao = ExternalAgentDao(snowpark_session=self.sf_session)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    def test_create_agent_agent_not_exists(self, mock_execute_query):
        empty_df = pd.DataFrame({"name": []})
        mock_execute_query.return_value = empty_df

        self.dao.create_agent_if_not_exist("agent1", "v1")

        expected_name = "agent1"
        expected_query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        expected_parameters = (expected_name, "v1")
        expected_message = (
            f"Created External Agent {expected_name} with version v1."
        )

        # create_new_agent is called, so sql_utils.execute_query should be called once.
        mock_execute_query.assert_any_call(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )

        self.assertEqual(mock_execute_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    def test_create_agent_agent_exists_version_exists(self, mock_execute_query):
        df_agents = pd.DataFrame({"name": ["DB.SCH.agent1"]})
        # For list_agent_versions, simulate that version "v1" already exists by returning a DataFrame.
        df_versions = pd.DataFrame({"version": ["v1"]})
        mock_execute_query.side_effect = [df_agents, df_versions]

        self.dao.create_agent_if_not_exist("agent1", "v1")

        # No query should be executed since the agent and version already exist.
        self.assertEqual(mock_execute_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    def test_create_agent_agent_exists_version_not_exists_empty_list(
        self, mock_execute_query
    ):
        # Simulate that the agent exists.
        df_agents = pd.DataFrame({"name": ["agent1"]})
        # Simulate that no versions exist.
        df_versions = pd.DataFrame({"version": []})
        # Then a creation call for adding the new version.
        df_creation = pd.DataFrame()
        # For the subsequent call to list_agent_versions, simulate that no version exists. This should be rare.
        mock_execute_query.side_effect = [df_agents, df_versions, df_creation]

        self.dao.create_agent_if_not_exist("agent1", "v2")

        expected_name = "agent1"
        expected_query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        expected_parameters = (expected_name, "v2")
        expected_message = (
            f"Added version v2 to External Agent {expected_name}."
        )

        # The DAO should call execute_query to add the new version.
        mock_execute_query.assert_any_call(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )
        self.assertEqual(mock_execute_query.call_count, 3)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    def test_create_agent_agent_exists_version_not_in_existing_list(
        self, mock_execute_query
    ):
        # agent exists and some versions are present,
        # but different from the version we want to add.
        df_agents = pd.DataFrame({"name": ["agent1"]})
        df_versions = pd.DataFrame({"version": ["v1", "v2"]})
        df_creation = pd.DataFrame()
        # Existing versions do not include "v3"
        mock_execute_query.side_effect = [df_agents, df_versions, df_creation]

        self.dao.create_agent_if_not_exist("agent1", "v3")

        expected_name = "agent1"
        expected_query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        expected_parameters = (expected_name, "v3")
        expected_message = (
            f"Added version v3 to External Agent {expected_name}."
        )

        # The DAO should call execute_query to add the new version.
        mock_execute_query.assert_any_call(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )
        self.assertEqual(mock_execute_query.call_count, 3)


if __name__ == "__main__":
    main()
