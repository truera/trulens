import logging
from typing import List

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class ExternalAgentDAO:
    """Data Access Object (DAO) for managing External Agents in Snowflake using qmark parameter binding."""

    def __init__(self, snowpark_session: Session):
        """Initialize with an active Snowpark session."""
        self.session = snowpark_session
        self.database = snowpark_session.get_current_database()
        self.schema = snowpark_session.get_current_schema()
        logger.info("Initialized ExternalAgentDAO with a Snowpark session.")

    def create_agent(self, name: str, version: str) -> None:
        """Create a new External Agent with a specified version."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "CREATE EXTERNAL AGENT ? WITH VERSION ?;"
        parameters = (agent_fqn, version)
        self._execute_query(
            query,
            parameters,
            f"Created External Agent {agent_fqn} with version {version}.",
        )

    def clone_agent(
        self,
        new_name: str,
        new_version: str,
        source_fqn: str,
        source_version: str,
    ) -> None:
        """Create a new External Agent from an existing one."""
        agent_fqn = f"{self.database}.{self.schema}.{new_name}"

        query = "CREATE EXTERNAL AGENT ? WITH VERSION ? FROM EXTERNAL AGENT ? VERSION ?;"
        parameters = (agent_fqn, new_version, source_fqn, source_version)
        self._execute_query(
            query,
            parameters,
            f"Cloned External Agent {new_name} from {source_fqn} (version {source_version}).",
        )

    def drop_agent(self, name: str) -> None:
        """Delete an External Agent."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "DROP EXTERNAL AGENT ?;"
        parameters = (agent_fqn,)
        self._execute_query(
            query, parameters, f"Dropped External Agent {agent_fqn}."
        )

    def add_version(self, name: str, version: str) -> None:
        """Add a new version to an existing External Agent."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "ALTER EXTERNAL AGENT ? ADD VERSION ?;"
        parameters = (agent_fqn, version)
        self._execute_query(
            query,
            parameters,
            f"Added version {version} to External Agent {agent_fqn}.",
        )

    def drop_version(self, name: str, version: str) -> None:
        """Drop a specific version from an External Agent."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "ALTER EXTERNAL AGENT ? DROP VERSION ?;"
        parameters = (agent_fqn, version)
        self._execute_query(
            query,
            parameters,
            f"Dropped version {version} from External Agent {agent_fqn}.",
        )

    def set_default_version(
        self,
        name: str,
        version: str,
    ) -> None:
        """Set the default version for an External Agent."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "ALTER EXTERNAL AGENT ? SET DEFAULT_VERSION = ?;"
        parameters = (agent_fqn, version)
        self._execute_query(
            query,
            parameters,
            f"Set default version {version} for External Agent {agent_fqn}.",
        )

    def list_agents(self) -> List[str]:
        """Retrieve a list of all External Agents."""
        query = "SHOW EXTERNAL AGENTS;"
        return self._fetch_query(
            query, "Retrieved list of External Agents.", "name", parameters=()
        )

    def list_versions(self, name: str) -> List[str]:
        """Retrieve all versions of a specific External Agent."""
        agent_fqn = f"{self.database}.{self.schema}.{name}"
        query = "SHOW VERSIONS IN EXTERNAL AGENT ?;"
        parameters = (agent_fqn,)
        return self._fetch_query(
            query,
            f"Retrieved versions for External Agent {agent_fqn}.",
            "version",
            parameters,
        )

    def _execute_query(
        self, query: str, parameters: tuple, success_message: str
    ) -> None:
        """Executes a query with parameters and logs the result, with error handling."""
        try:
            # Execute the SQL with parameter binding using qmark style
            self.session.sql(query, params=parameters).collect()
            logger.info(success_message)
        except Exception as e:
            logger.error(
                f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
            )
            raise RuntimeError(f"Failed to execute query: {query}") from e

    def _fetch_query(
        self,
        query: str,
        success_message: str,
        field_name: str,
        parameters: tuple,
    ) -> List[str]:
        """Executes a query and returns a list of values from a specified field."""
        try:
            if parameters:
                result = self.session.sql(query, params=parameters).collect()
            else:
                result = self.session.sql(query).collect()
            logger.info(success_message)
            return [row[field_name] for row in result]
        except Exception as e:
            logger.error(
                f"Error fetching query: {query}\nParameters: {parameters}\nError: {e}"
            )
            raise RuntimeError(f"Failed to fetch query: {query}") from e
