import logging

import pandas
from snowflake.snowpark import Session
from trulens.connectors.snowflake.dao.sql_utils import execute_query

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake."""

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def create_new_agent(self, name: str, version: str) -> None:
        """Create a new External Agent with a specified version."""

        query = f"CREATE EXTERNAL AGENT {name} WITH VERSION {version};"
        execute_query(self.session, query)

        logger.info(f"Created External Agent {name} with version {version}.")

    def create_agent_if_not_exist(self, name: str, version: str) -> None:
        """
        Args:
            name (str): unique name of the external agent
            version (str): version is mandatory for now
        """
        # Get the agent if it already exists, otherwise create it

        if not self.check_agent_exists(name):
            self.create_new_agent(name, version)
        else:
            # Check if the version exists for the agent
            existing_versions = self.list_agent_versions(name)

            if (
                existing_versions.empty
                or "name" not in existing_versions.columns
                or version.upper() not in existing_versions["name"].values
            ):
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

        query = "DROP EXTERNAL AGENT IDENTIFIER(?);"
        parameters = (name,)
        execute_query(self.session, query, parameters)

        logger.info(f"Dropped External Agent {name}.")

    def add_version(self, name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""

        query = f"ALTER EXTERNAL AGENT if exists {name}  ADD VERSION {version};"
        # parameter bindings doesn't work with ALTER statement

        execute_query(self.session, query)

        logger.info(f"Added version {version} to External Agent {name}.")

    def drop_version(self, name: str, version: str) -> None:
        """Drop a specific version from an External Agent."""

        query = f"ALTER EXTERNAL AGENT if exists {name} DROP VERSION {version};"

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

        return name.upper() in agents["name"].values

    def list_agent_versions(self, name: str) -> pandas.DataFrame:
        """Retrieve all versions of a specific External Agent."""

        query = "SHOW VERSIONS IN EXTERNAL AGENT IDENTIFIER(?);"
        parameters = (name,)

        rows = execute_query(self.session, query, parameters)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info(f"Retrieved versions for External Agent {name}.")

        return result_df
