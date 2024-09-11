from trulens.dashboard import run_dashboard
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
import os

connection_params = {
    "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
    "user": os.environ.get("SNOWFLAKE_USER"),
    "password": os.environ.get("SNOWFLAKE_PASSWORD"),
    "database": os.environ.get("SNOWFLAKE_DATABASE"),
    "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
    "role": os.environ.get("SNOWFLAKE_ROLE"),
    "init_server_side": False,
}

connector = SnowflakeConnector(**connection_params)
session = TruSession(connector=connector, init_server_side=False)

run_dashboard(session, port=8484)