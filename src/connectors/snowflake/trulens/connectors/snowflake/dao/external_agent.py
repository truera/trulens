import logging
from typing import List

import pandas
from snowflake.snowpark import Session
import trulens.connectors.snowflake.dao.sql_utils as sql_utils

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake."""

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session = snowpark_session
        self.database = snowpark_session.get_current_database()
        self.schema = snowpark_session.get_current_schema()
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def _get_agent_fqn(self, name: str) -> str:
        """Return the fully qualified name (FQN) for an External Agent."""
        return f"{self.database}.{self.schema}.{name}"

    def resolve_agent_name(self, name: str) -> str:
        """
        Resolve the agent name into a fully qualified name.
        If the provided name already appears fully qualified (e.g. it has three parts), return it as is.
        Otherwise, use _get_agent_fqn to create a FQN.
        """
        parts = name.split(".")
        # Assuming a fully qualified name has exactly three parts: database, schema, and object name.
        if len(parts) == 3:
            return name
        return self._get_agent_fqn(name)

    def create_new_agent(self, name: str, version: str) -> None:
        """Create a new External Agent with a specified version."""
        agent_fqn = self.resolve_agent_name(name)
        query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        parameters = (agent_fqn, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Created External Agent {agent_fqn} with version {version}.",
        )

    def create_agent_if_not_exist(self, name: str, version: str) -> None:
        # Get the agent if it already exists, otherwise create it
        agent_fqn = self.resolve_agent_name(name)
        if agent_fqn not in self.list_agents()["name"].values:
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
        agent_fqn = self.resolve_agent_name(name)
        query = "DROP EXTERNAL AGENT ?;"
        parameters = (agent_fqn,)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Dropped External Agent {agent_fqn}.",
        )

    def add_version(self, name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""
        agent_fqn = self.resolve_agent_name(name)
        query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        parameters = (agent_fqn, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Added version {version} to External Agent {agent_fqn}.",
        )

    def drop_version(self, name: str, version: str) -> None:
        """Drop a specific version from an External Agent."""
        agent_fqn = self.resolve_agent_name(name)
        query = "ALTER EXTERNAL AGENT ? DROP VERSION ?;"
        parameters = (agent_fqn, version)
        sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Dropped version {version} from External Agent {agent_fqn}.",
        )

    def list_agents(self) -> pandas.DataFrame:
        """Retrieve a list of all External Agents."""
        query = "SHOW EXTERNAL AGENTS;"
        return sql_utils.fetch_query(
            self.session,
            query,
            "Retrieved list of External Agents.",
            parameters=(),
        )

    def list_agent_versions(self, name: str) -> List[str]:
        """Retrieve all versions of a specific External Agent."""
        agent_fqn = self.resolve_agent_name(name)
        query = "SHOW VERSIONS IN EXTERNAL AGENT ?;"
        parameters = (agent_fqn,)
        result_df = sql_utils.fetch_query(
            self.session,
            query,
            f"Retrieved versions for External Agent {agent_fqn}.",
            parameters,
        )

        return result_df["version"].values.tolist()
