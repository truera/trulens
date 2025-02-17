import logging
from typing import Tuple

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

    def create_new_agent(self, name: str, version: str) -> Tuple[str, str]:
        """Create a new External Agent with a specified version."""

        query = f"CREATE EXTERNAL AGENT {name} WITH VERSION {version};"
        execute_query(self.session, query)

        logger.info(f"Created External Agent {name} with version {version}.")
        return name, version

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

    def list_agent_versions(self, name: str) -> pandas.DataFrame:
        """Retrieve all versions of a specific External Agent."""

        query = "SHOW VERSIONS IN EXTERNAL AGENT IDENTIFIER(?);"
        parameters = (name,)

        rows = execute_query(self.session, query, parameters)
        result_df = pandas.DataFrame([row.as_dict() for row in rows])

        logger.info(f"Retrieved versions for External Agent {name}.")

        return result_df
