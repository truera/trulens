import logging
from typing import List, Optional

from snowflake.snowpark import Row
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

DOUBLE_QUOTE = '"'


def clean_up_snowflake_identifier(
    snowflake_identifier: Optional[str],
) -> Optional[str]:
    if not snowflake_identifier:
        return snowflake_identifier
    if snowflake_identifier[0] == '"' and snowflake_identifier[-1] == '"':
        return snowflake_identifier[1:-1]
    return snowflake_identifier


def escape_quotes(unescaped: str) -> str:
    """Escapes double quotes in a string by doubling them.

    Args:
        unescaped (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    return unescaped.replace(DOUBLE_QUOTE, DOUBLE_QUOTE + DOUBLE_QUOTE)


def double_quote_identifier(identifier: str) -> str:
    """Double quotes the identifier to preserve it as-is in SQL.

    Args:
        identifier (str): The identifier to double quote.

    Returns:
        str: The double quoted identifier.
    """
    return DOUBLE_QUOTE + escape_quotes(identifier) + DOUBLE_QUOTE


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
