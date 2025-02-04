import sys
from unittest import main
from unittest import skipIf
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd

from tests.test import TruTestCase
from tests.test import optional_test
from tests.test import run_optional_tests

if run_optional_tests():
    from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao


@skipIf(
    sys.version_info >= (3, 12),
    "trulens-connector-snowflake is not yet supported in Python 3.12",
)
@optional_test
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
    @patch("trulens.connectors.snowflake.dao.sql_utils.fetch_query")
    def test_create_agent_agent_not_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        empty_df = pd.DataFrame({"name": []})
        mock_fetch_query.return_value = empty_df

        self.dao.create_agent_if_not_exist("agent1", "v1")

        expected_fqn = "DB.SCH.agent1"
        expected_query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        expected_parameters = (expected_fqn, "v1")
        expected_message = (
            f"Created External Agent {expected_fqn} with version v1."
        )

        # create_new_agent is called, so sql_utils.execute_query should be called once.
        mock_execute_query.assert_called_once_with(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )
        # list_agents should be called once.
        mock_fetch_query.assert_called_once()

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    @patch("trulens.connectors.snowflake.dao.sql_utils.fetch_query")
    def test_create_agent_agent_exists_version_exists(
        self, mock_fetch_query, mock_execute_query
    ):
        df_agents = pd.DataFrame({"name": ["DB.SCH.agent1"]})
        # For the subsequent call to list_agent_versions, simulate that version "v1" already exists.
        mock_fetch_query.side_effect = [df_agents, ["v1"]]

        self.dao.create_agent_if_not_exist("agent1", "v1")

        # No query should be executed since the agent and version already exist.
        mock_execute_query.assert_not_called()
        self.assertEqual(mock_fetch_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    @patch("trulens.connectors.snowflake.dao.sql_utils.fetch_query")
    def test_create_agent_agent_exists_version_not_exists_empty_list(
        self, mock_fetch_query, mock_execute_query
    ):
        df_agents = pd.DataFrame({"name": ["DB.SCH.agent1"]})
        # For the subsequent call to list_agent_versions, simulate that no version exists. This should be rare.
        mock_fetch_query.side_effect = [df_agents, []]

        self.dao.create_agent_if_not_exist("agent1", "v2")

        expected_fqn = "DB.SCH.agent1"
        expected_query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        expected_parameters = (expected_fqn, "v2")
        expected_message = f"Added version v2 to External Agent {expected_fqn}."

        # The DAO should call execute_query to add the new version.
        mock_execute_query.assert_called_once_with(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )
        self.assertEqual(mock_fetch_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    @patch("trulens.connectors.snowflake.dao.sql_utils.fetch_query")
    def test_create_agent_agent_exists_version_not_in_existing_list(
        self, mock_fetch_query, mock_execute_query
    ):
        # agent exists and some versions are present,
        # but different from the version we want to add.
        df_agents = pd.DataFrame({"name": ["DB.SCH.agent1"]})
        # Existing versions do not include "v3"
        mock_fetch_query.side_effect = [df_agents, ["v1", "v2"]]

        self.dao.create_agent_if_not_exist("agent1", "v3")

        expected_fqn = "DB.SCH.agent1"
        expected_query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        expected_parameters = (expected_fqn, "v3")
        expected_message = f"Added version v3 to External Agent {expected_fqn}."

        # The DAO should call execute_query to add the new version.
        mock_execute_query.assert_called_once_with(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )
        self.assertEqual(mock_fetch_query.call_count, 2)

    @patch("trulens.connectors.snowflake.dao.sql_utils.execute_query")
    def test_create_new_agent_with_fqn(self, mock_execute_query):
        """
        Test that if a fully qualified agent name is provided,
        create_new_agent uses it unchanged.
        """
        # Provide a fully qualified agent name.
        fully_qualified_name = "MYDB.MYSC.agentX"
        version = "v1"
        self.dao.create_new_agent(fully_qualified_name, version)

        expected_query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        expected_parameters = (fully_qualified_name, version)
        expected_message = f"Created External Agent {fully_qualified_name} with version {version}."

        mock_execute_query.assert_called_once_with(
            self.sf_session,
            expected_query,
            expected_parameters,
            expected_message,
        )


if __name__ == "__main__":
    main()
