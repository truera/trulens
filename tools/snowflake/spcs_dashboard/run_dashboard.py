import os

from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession
from trulens.dashboard import run_dashboard


def get_login_token():
    """
    Read the login token supplied automatically by Snowflake. These tokens
    are short lived and should always be read right before creating any new connection.
    """
    with open("/snowflake/session/token", "r") as f:
        return f.read()


connection_params = {
    "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
    "host": os.getenv("SNOWFLAKE_HOST"),
    "authenticator": "oauth",
    "token": get_login_token(),
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
    "database": os.environ.get("SNOWFLAKE_DATABASE"),
    "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
}
snowpark_session = Session.builder.configs(connection_params).create()

connector = SnowflakeConnector(snowpark_session=snowpark_session)
tru_session = TruSession(connector=connector)
tru_session.get_records_and_feedback()

run_dashboard(tru_session, port=8484, spcs_runtime=True)
