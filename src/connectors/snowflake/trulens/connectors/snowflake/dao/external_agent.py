import logging
from typing import Tuple

import pandas
from snowflake.snowpark import Session
from trulens.connectors.snowflake.dao.sql_utils import escape_quotes
from trulens.connectors.snowflake.dao.sql_utils import execute_query

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake.
    We currently enclose all names in double quotes to preserve the passed in string as-is when converting to SQL identifiers.
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    double quotes in the passed in string will be escaped with an additional double quote
    """

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def create_new_agent(self, name: str, version: str) -> Tuple[str, str]:
        """Create a new External Agent with a specified version."""
        escaped_name = escape_quotes(name)  # escape double quotes
        # note we cannot parametrize inputs to query when using CREATE statement - hence f-string but there might be a risk of sql injection
        query = f"""CREATE EXTERNAL AGENT IDENTIFIER('"{escaped_name}"') WITH VERSION "{version}";"""
        execute_query(self.session, query)

        logger.info(f"Created External Agent {name} with version {version}.")

        return name, version

    def create_agent_if_not_exist(
        self, name: str, version: str
    ) -> Tuple[str, str]:
        """
        Args:
            name (str): unique name of the external agent
            version (str): version is mandatory for now
        Returns:
            Tuple[str, str]: resolved_name, version
        """
        # Get the agent if it already exists, otherwise create it
        logger.error(self.check_agent_exists(name))

        if not self.check_agent_exists(name):
            self.create_new_agent(name, version)
        else:
            # Check if the version exists for the agent
            existing_versions = self.list_agent_versions(name)

            if (
                existing_versions.empty
                or "name" not in existing_versions.columns
                or version not in existing_versions["name"].values
            ):
                self.add_version(name, version)
                logger.info(
                    f"Added version {version} to External Agent {name}."
                )
            else:
                logger.info(
                    f"External Agent {name} with version {version} already exists."
                )
        return name, version

    def drop_agent(self, name: str) -> None:
        """Delete an External Agent."""
        escaped_name = escape_quotes(name)
        query = f"""DROP EXTERNAL AGENT IDENTIFIER('"{escaped_name}"');"""

        execute_query(self.session, query)

        logger.info(f"Dropped External Agent {name}.")

    def add_version(self, name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""
        escaped_name = escape_quotes(name)
        query = f"""ALTER EXTERNAL AGENT if exists IDENTIFIER('"{escaped_name}"')  ADD VERSION "{version}";"""
        # parameter bindings doesn't work with ALTER statement

        execute_query(self.session, query)

        logger.info(f"Added version {version} to External Agent {name}.")

    def drop_version(self, name: str, version: str) -> None:
        """Drop a specific version from an External Agent."""
        escaped_name = escape_quotes(name)
        query = f"""ALTER EXTERNAL AGENT if exists IDENTIFIER('"{escaped_name}"') DROP VERSION "{version}";"""

        execute_query(self.session, query)
        logger.info(f"Dropped version {version} from External Agent {name}.")

    def list_agents(self) -> pandas.DataFrame:
        """Retrieve a list of all External Agents."""
        query = "SHOW EXTERNAL AGENTS;"

        rows = execute_query(self.session, query)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info("Retrieved list of External Agents.")
        return result_df

    def check_agent_exists(self, name: str) -> bool:
        """Check if an External Agent exists."""
        agents = self.list_agents()
        if agents.empty or "name" not in agents.columns:
            return False
        logger.info(f"Checking if External Agent {name} exists.")

        return name in agents["name"].values

    def list_agent_versions(self, name: str) -> pandas.DataFrame:
        """Retrieve all versions of a specific External Agent."""
        escaped_name = escape_quotes(name)

        query = f"""SHOW VERSIONS IN EXTERNAL AGENT IDENTIFIER('"{escaped_name}"');"""

        rows = execute_query(self.session, query)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info(f"Retrieved versions for External Agent {name}.")

        return result_df
