import argparse
import sys
import os
from trulens.core import TruSession
from trulens.core.database import base as mod_db
from snowflake.snowpark import Session
from snowflake.sqlalchemy import URL


def init_from_args():
    """Parse command line arguments and initialize Tru with them.

    As Tru is a singleton, further TruSession() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--database-prefix", default=mod_db.DEFAULT_DATABASE_PREFIX
    )
    parser.add_argument("--spcs-runtime", default=False)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

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
    db_url = URL(
        account=snowpark_session.get_current_account(),
        user=snowpark_session.get_current_user(),
        password="password",
        database=snowpark_session.get_current_database(),
        schema=snowpark_session.get_current_schema(),
        warehouse=snowpark_session.get_current_warehouse(),
        role=snowpark_session.get_current_role(),
    )
    if args.spcs_runtime:
        TruSession(
            database_url=db_url,
            database_check_revision=False,
            database_args=database_args,
            database_prefix=args.database_prefix
        )
    else:
        TruSession(
            database_url=args.database_url, database_prefix=args.database_prefix
        )
