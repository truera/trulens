import logging
from typing import Optional

import pandas
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def execute_query(
    session: Session,
    query: str,
    parameters: Optional[tuple] = None,
) -> pandas.DataFrame:
    """
    Executes a query with optional parameters with qmark parameter binding (if applicable).
    """
    try:
        return session.sql(query, params=parameters).collect()
    except Exception as e:
        logger.exception(
            f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise RuntimeError(f"Failed to execute query: {query}") from e
