from trulens.dashboard import run_dashboard
from trulens.core import TruSession
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from snowflake.sqlalchemy import URL
import os

# connection_params = {
#     "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
#     "host": os.getenv("SNOWFLAKE_HOST"),
#     "user": os.environ.get("SNOWFLAKE_USER"),
#     "password": os.environ.get("SNOWFLAKE_PASSWORD"),
#     "database": os.environ.get("SNOWFLAKE_DATABASE"),
#     "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
#     "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
#     "role": os.environ.get("SNOWFLAKE_ROLE"),
#     "init_server_side": False,
# }

# connector = SnowflakeConnector(**connection_params)
# session = TruSession(connector=connector, init_server_side=False)

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

# Set up sqlalchemy engine parameters.
conn = snowpark_session.connection
engine_params = {}
engine_params["paramstyle"] = "qmark"
engine_params["creator"] = lambda: conn
database_args = {"engine_params": engine_params}
# # Ensure any Cortex provider uses the only Snowflake connection allowed in this stored procedure.
# trulens.providers.cortex.provider._SNOWFLAKE_STORED_PROCEDURE_CONNECTION = (
#     conn
# )
# Run deferred feedback evaluator.
db_url = URL(
    account=snowpark_session.get_current_account(),
    user=snowpark_session.get_current_user(),
    password="password",
    database=snowpark_session.get_current_database(),
    schema=snowpark_session.get_current_schema(),
    warehouse=snowpark_session.get_current_warehouse(),
    role=snowpark_session.get_current_role(),
)
tru_session = TruSession(
    database_url=db_url,
    database_check_revision=False,  # TODO: check revision in the future?
    database_args=database_args,
)
tru_session.get_records_and_feedback()

run_dashboard(tru_session, port=8484, spcs_runtime=True)