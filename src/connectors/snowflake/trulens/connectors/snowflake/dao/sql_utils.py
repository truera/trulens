import logging

import pandas
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def execute_query(
    session: Session, query: str, parameters: tuple, success_message: str
) -> None:
    """Executes a query with parameters and logs the result, with error handling."""
    try:
        # Execute the SQL with parameter binding using qmark style
        session.sql(query, params=parameters).collect()
        logger.info(success_message)
    except Exception as e:
        logger.error(
            f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise RuntimeError(f"Failed to execute query: {query}") from e


def fetch_query(
    session: Session,
    query: str,
    success_message: str,
    parameters: tuple,
) -> pandas.DataFrame:
    """Executes a query and returns a list of values from a specified field."""
    try:
        if parameters:
            result_df = session.sql(query, params=parameters).to_pandas()
        else:
            result_df = session.sql(query).to_pandas()
        logger.info(success_message)
        return result_df
    except Exception as e:
        logger.error(
            f"Error fetching query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise RuntimeError(f"Failed to fetch query: {query}") from e
