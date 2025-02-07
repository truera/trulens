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

    def _quote_if_needed(self, identifier: str) -> str:
        """
        If the identifier is already quoted, return it unchanged.
        If the identifier matches the simple unquoted identifier pattern (i.e.
        starts with a letter (A-Z, a-z) or an underscore, and contains only letters,
        decimal digits (0-9), underscores, or dollar signs),
        return the identifier in uppercase (since Snowflake stores unquoted identifiers as uppercase).
        Otherwise, wrap the identifier in double quotes.
        ref: https://docs.snowflake.com/en/sql-reference/identifiers-syntax
        """
        # If already double-quoted, return as-is.
        if identifier.startswith('"') and identifier.endswith('"'):
            return identifier

        # If it matches the simple identifier pattern, return in uppercase.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", identifier):
            return identifier.upper()

        # Otherwise, wrap it in double quotes.
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

    def create_agent_if_not_exist(self, name: str, version: str) -> str:
        """
        Args:
            name (str): unique name of the external agent
            version (str): version is mandatory for now
        Returns:
            str: fully qualified name of the external agent
        """
        # Get the agent if it already exists, otherwise create it
        new_agent_fqn = self.resolve_agent_name(name)
        if new_agent_fqn not in self.list_agents()["name"].values:
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
        return new_agent_fqn

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
        return sql_utils.execute_query(
            self.session,
            query,
            parameters=(),
            success_message="Retrieved list of External Agents.",
        )

    def list_agent_versions(self, name: str) -> List[str]:
        """Retrieve all versions of a specific External Agent."""
        agent_fqn = self.resolve_agent_name(name)
        query = "SHOW VERSIONS IN EXTERNAL AGENT ?;"
        parameters = (agent_fqn,)
        result_df = sql_utils.execute_query(
            self.session,
            query,
            parameters,
            f"Retrieved versions for External Agent {agent_fqn}.",
        )

        return result_df["version"].values.tolist()

    def check_agent_exists(self, name: str) -> bool:
        """Check if an External Agent exists."""
        agent_fqn = self.resolve_agent_name(name)
        return agent_fqn in self.list_agents()["name"].values
