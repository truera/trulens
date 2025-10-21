import logging
from typing import Optional

import pandas as pd
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
) -> pd.DataFrame:
    """
    Executes a query with optional parameters with qmark parameter binding (if applicable).
    """
    try:
        if query.strip().upper().startswith("SELECT"):
            # snowpark to_pandas supports only SELECT statements up until recent releases https://github.com/snowflakedb/snowpark-python/blob/main/CHANGELOG.md
            df = session.sql(query, params=parameters).to_pandas()
        else:
            result = session.sql(query, params=parameters).collect()
            if not result:
                return pd.DataFrame()
            # Use actual field names from the Row objects
            columns = result[0]._fields
            data = [tuple(row) for row in result]
            df = pd.DataFrame(data, columns=columns)

        df.columns = [clean_up_snowflake_identifier(col) for col in df.columns]
        return df
    except Exception as e:
        logger.exception(
            f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise
