import logging
import re
from typing import List, Optional

from snowflake.snowpark import Row
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_SF_UNQUOTED_CASE_INSENSITIVE_IDENTIFIER = "[A-Za-z_][A-Za-z0-9_$]*"
_SF_UNQUOTED_CASE_SENSITIVE_IDENTIFIER = "[A-Z_][A-Z0-9_$]*"
SF_QUOTED_IDENTIFIER = '"(?:[^"]|"")*"'
QUOTED_IDENTIFIER_RE = re.compile(f"^({SF_QUOTED_IDENTIFIER})$")
UNQUOTED_CASE_INSENSITIVE_RE = re.compile(
    f"^({_SF_UNQUOTED_CASE_INSENSITIVE_IDENTIFIER})$"
)

UNQUOTED_CASE_SENSITIVE_RE = re.compile(
    f"^({_SF_UNQUOTED_CASE_SENSITIVE_IDENTIFIER})$"
)
DOUBLE_QUOTE = '"'


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
        raise RuntimeError(f"Failed to execute query: {query}") from e


# logic borrowed from https://github.com/snowflakedb/snowml/blob/main/snowflake/ml/_internal/utils/identifier.py
# we could consider directly import when snowflake-ml-python becomes required in trulens-connectors-snowflake
def _is_quoted(id: str) -> bool:
    """Checks if input *identifier* is quoted.

    NOTE: Snowflake treats all identifiers as UPPERCASE by default. That is 'Hello' would become 'HELLO'. To preserve
    case, one needs to use quoted identifiers, e.g. "Hello" (note the double quote). Callers must take care of that
    quoting themselves. This library assumes that if there is double-quote both sides, it is escaped, otherwise does not
    require.

    Args:
        id: The string to be checked

    Returns:
        True if the `id` is quoted with double-quote to preserve case. Returns False otherwise.

    Raises:
        ValueError: If the id is invalid.
    """
    if not id:
        raise ValueError(f"Invalid id {id} passed. ID is empty.")
    if len(id) >= 2 and id[0] == '"' and id[-1] == '"':
        if len(id) == 2:
            raise ValueError(f"Invalid id {id} passed. ID is empty.")
        if not QUOTED_IDENTIFIER_RE.match(id):
            raise ValueError(
                f"Invalid id {id} passed. ID is quoted but does not match the quoted rule."
            )
        return True
    if not UNQUOTED_CASE_SENSITIVE_RE.match(id):
        raise ValueError(
            f"Invalid id {id} passed. ID is unquoted but does not match the unquoted rule."
        )
    return False


def _get_unescaped_name(id: str) -> str:
    """Remove double quotes and unescape quotes between them from id if quoted.
        Return as it is otherwise

    NOTE: See note in `_is_quoted`.

    Args:
        id: The string to be checked & treated.

    Returns:
        String with quotes removed if quoted; original string otherwise.
    """
    if not _is_quoted(id):
        return id
    unquoted_id = id[1:-1]
    return unquoted_id.replace(DOUBLE_QUOTE + DOUBLE_QUOTE, DOUBLE_QUOTE)


def resolve_identifier(name: str) -> str:
    """Given a user provided *string*, resolve following Snowflake identifier resolution strategies:
        https://docs.snowflake.com/en/sql-reference/identifiers-syntax#label-identifier-casing

        This function will mimic the behavior of the SQL parser.

    Examples:
        COL1 -> COL1
        1COL -> Raise Error
        Col -> COL
        "COL" -> COL
        COL 1 -> Raise Error

    Args:
        name: the string to be resolved.

    Raises:
        ValueError: if input would not be accepted by SQL parser.

    Returns:
        Resolved identifier
    """
    if QUOTED_IDENTIFIER_RE.match(name):
        unescaped = _get_unescaped_name(name)
        if UNQUOTED_CASE_SENSITIVE_RE.match(unescaped):
            return unescaped
        return name
    elif UNQUOTED_CASE_INSENSITIVE_RE.match(name):
        return name.upper()
    else:
        raise ValueError(
            f"{name} is not a valid SQL identifier: https://docs.snowflake.com/en/sql-reference/identifiers-syntax"
        )
