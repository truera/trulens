import logging

import pandas
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def execute_query(
    session: Session,
    query: str,
    parameters: tuple,
    success_message: str,
) -> pandas.DataFrame:
    """
    Executes a query with parameters with qmark parameter binding.
    """
    try:
        sql_obj = session.sql(query, params=parameters)
        result_df = sql_obj.to_pandas()
        logger.info(success_message)
        return result_df
    except Exception as e:
        logger.error(
            f"Error executing query: {query}\nParameters: {parameters}\nError: {e}"
        )
        raise RuntimeError(f"Failed to execute query: {query}") from e
