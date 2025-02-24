import logging
from typing import List, Optional

from snowflake.snowpark import Row
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

DOUBLE_QUOTE = '"'


def escape_quotes(unescaped: str) -> str:
    """Escapes double quotes in a string by doubling them.

    Args:
        unescaped (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    return unescaped.replace(DOUBLE_QUOTE, DOUBLE_QUOTE + DOUBLE_QUOTE)


def execute_query(
    session: Session,
    query: str,
    parameters: Optional[tuple] = None,
) -> List[Row]:
    """
    Executes a query with optional parameters with qmark parameter binding (if applicable).
    """
    try:
        return session.sql(query, params=parameters).collect()
    except Exception as e:
        logger.exception(
            f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise
