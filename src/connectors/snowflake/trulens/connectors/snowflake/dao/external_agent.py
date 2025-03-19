import logging
from typing import Tuple

import pandas
from snowflake.snowpark import Session
from trulens.connectors.snowflake.dao.sql_utils import execute_query

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake.
    We currently use unquoted object identifiers when converting to SQL identifiers.
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    """

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def _create_new_agent(self, resolved_name: str, version: str) -> None:
        """Create a new External Agent with a specified version."""

        # note we cannot parametrize inputs to query when using CREATE statement - hence f-string but there might be a risk of sql injection
        query = f"""CREATE EXTERNAL AGENT "{resolved_name}" WITH VERSION "{version}";"""
        execute_query(self.session, query)

        logger.info(
            f"Created External Agent {resolved_name} with version {version}."
        )

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
        resolved_name = name.upper()
        # Get the agent if it already exists, otherwise create it
        if not self.check_agent_exists(resolved_name):
            self._create_new_agent(resolved_name, version)
        else:
            # Check if the version exists for the agent
            existing_versions = self.list_agent_versions(resolved_name)

            if (
                existing_versions.empty
                or "name" not in existing_versions.columns
                or version not in existing_versions["name"].values
            ):
                self.add_version(resolved_name, version)
                logger.info(
                    f"Added version {version} to External Agent {resolved_name}."
                )
            else:
                logger.info(
                    f"External Agent {resolved_name} with version {version} already exists."
                )

        return resolved_name, version

    def drop_agent(self, name: str) -> None:
        """Delete an External Agent."""
        resolved_name = name.upper()
        query = f"""DROP EXTERNAL AGENT "{resolved_name}";"""

        execute_query(self.session, query)

        logger.info(f"Dropped External Agent {name}.")

    def add_version(self, resolved_name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""
        query = f"""ALTER EXTERNAL AGENT if exists "{resolved_name}"  ADD VERSION "{version}";"""
        # parameter bindings doesn't work with ALTER statement

        execute_query(self.session, query)

        logger.info(
            f"Added version {version} to External Agent {resolved_name}."
        )

    def _get_current_version(self, name: str) -> str:
        """Retrieve the current version of an External Agent."""
        versions_df = self.list_agent_versions(name)

        if not versions_df.empty:
            for _, row in versions_df.iterrows():
                if "LAST" in row["aliases"]:
                    current_version = row["name"]
                    return current_version

        raise ValueError(f"No versions found for External Agent {name}.")

    def drop_current_version(self, name: str) -> None:
        """Drop a specific version from an External Agent."""

        current_version = self._get_current_version(name)
        resolved_name = name.upper()
        query = f"""ALTER EXTERNAL AGENT if exists "{resolved_name}" DROP VERSION "{current_version}";"""

        execute_query(self.session, query)
        logger.info(
            f"Dropped current version {current_version} from External Agent {name}."
        )

    def _list_agents(self) -> pandas.DataFrame:
        """Retrieve a list of all External Agents."""
        query = "SHOW EXTERNAL AGENTS;"

        rows = execute_query(self.session, query)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info("Retrieved list of External Agents.")
        return result_df

    def check_agent_exists(self, name: str) -> bool:
        """Check if an External Agent exists."""
        agents = self._list_agents()
        if agents.empty or "name" not in agents.columns:
            return False

        resolved_name = name.upper()
        logger.info(f"Checking if External Agent {resolved_name} exists.")

        return resolved_name in agents["name"].values

    def list_agent_versions(self, name: str) -> pandas.DataFrame:
        """Retrieve all versions of a specific External Agent."""
        resolved_name = name.upper()
        query = f"""SHOW VERSIONS IN EXTERNAL AGENT "{resolved_name}";"""

        rows = execute_query(self.session, query)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info(f"Retrieved versions for External Agent {resolved_name}.")

        return result_df
