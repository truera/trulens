import logging
import re
from typing import List

import pandas
from snowflake.snowpark import Session
import trulens.connectors.snowflake.dao.sql_utils as sql_utils

logger = logging.getLogger(__name__)


class ExternalAgentDao:
    """Data Access Object (DAO) layer for managing External Agents in Snowflake."""

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session
        self.database: str = snowpark_session.get_current_database()
        self.schema: str = snowpark_session.get_current_schema()
        logger.info("Initialized ExternalAgentDao with a Snowpark session.")

    def _get_agent_fqn(self, name: str) -> str:
        """Return the fully qualified name (FQN) for an External Agent."""
        return f"{self.database}.{self.schema}.{name}"

    def _quote_if_needed(self, identifier: str) -> str:
        """
        Note we only use qmark style parameter binding in our Snowflake connector.
        Return the identifier wrapped in double quotes if it does not match the pattern
        for a simple unquoted identifier in Snowflake. If the identifier is already quoted,
        return it unchanged.
        """
        if identifier.startswith('"') and identifier.endswith('"'):
            return identifier
        # A simple unquoted identifier in Snowflake must be all uppercase/lowercase letters, digits, dollar sign, or underscores.
        # https://docs.snowflake.com/en/sql-reference/identifiers-syntax
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", identifier):
            return identifier
        return f'"{identifier}"'

    def resolve_agent_name(self, name: str) -> str:
        """
        Resolve the agent name into a fully qualified name.
        If the provided name is already fully qualified (three dot-separated parts), return directly.
        Otherwise, construct the FQN from the current database and schema, quoting each part if needed.
        """
        parts = name.split(".")
        # Assuming a fully qualified name has exactly three parts: database, schema, and object name.
        if len(parts) == 3:
            return ".".join(self._quote_if_needed(part) for part in parts)
        return (
            f"{self._quote_if_needed(self.database)}."
            f"{self._quote_if_needed(self.schema)}."
            f"{self._quote_if_needed(name)}"
        )

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
        """
        Args:
            name (str): unique name of the external agent
            version (str): version is mandatory for now
        """
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
