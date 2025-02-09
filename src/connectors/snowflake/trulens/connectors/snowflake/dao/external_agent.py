import logging

import pandas
from snowflake.snowpark import Session
import trulens.connectors.snowflake.dao.sql_utils as sql_utils

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake."""

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def create_new_agent(self, name: str, version: str) -> None:
        """Create a new External Agent with a specified version."""

        query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        parameters = (name, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Created External Agent {name} with version {version}.",
        )

    def create_agent_if_not_exist(self, name: str, version: str) -> None:
        """
        Args:
            name (str): unique name of the external agent
            version (str): version is mandatory for now
        Returns:
            str: fully qualified name of the external agent
        """
        # Get the agent if it already exists, otherwise create it

        if name not in self.list_agents()["name"].values:
            self.create_new_agent(name, version)
        else:
            # Check if the version exists for the agent
            existing_versions = self.list_agent_versions(name)
            if version not in existing_versions:
                self.add_version(name, version)
                logger.info(
                    f"Added version {version} to External Agent {name}."
                )
            else:
                logger.info(
                    f"External Agent {name} with version {version} already exists."
                )

    def drop_agent(self, name: str) -> None:
        """Delete an External Agent."""

        query = "DROP EXTERNAL AGENT ?;"
        parameters = (name,)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Dropped External Agent {name}.",
        )

    def add_version(self, name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""

        query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        parameters = (name, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Added version {version} to External Agent {name}.",
        )

    def drop_version(self, name: str, version: str) -> None:
        """Drop a specific version from an External Agent."""

        query = "ALTER EXTERNAL AGENT ? DROP VERSION ?;"
        parameters = (name, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Dropped version {version} from External Agent {name}.",
        )

    def list_agents(self) -> pandas.DataFrame:
        """Retrieve a list of all External Agents."""
        query = "SHOW EXTERNAL AGENTS;"
        return sql_utils.execute_query(
            self.session,
            query,
            parameters=(),
            success_message="Retrieved list of External Agents.",
        )

    def list_agent_versions(self, name: str) -> pandas.DataFrame:
        """Retrieve all versions of a specific External Agent."""

        query = "SHOW VERSIONS IN EXTERNAL AGENT ?;"
        parameters = (name,)
        return sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Retrieved versions for External Agent {name}.",
        )

    def check_agent_exists(self, name: str) -> bool:
        """Check if an External Agent exists."""

        return name in self.list_agents()["name"].values
