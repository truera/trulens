import os

from snowflake.snowpark import Session

SNOWFLAKE_CONNECTION_NAME = os.environ.get("SNOWFLAKE_CONNECTION_NAME")


def _get_connection_params() -> dict:
    """Build params from env vars if set, otherwise use connection_name."""
    if os.environ.get("SNOWFLAKE_ACCOUNT"):
        return {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ.get("SNOWFLAKE_DATABASE", "SUPPORT_INTELLIGENCE"),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA", "DATA"),
            "role": os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "SUPPORT_INTELLIGENCE_WH"),
        }
    return {
        "connection_name": SNOWFLAKE_CONNECTION_NAME,
        "database": "SUPPORT_INTELLIGENCE",
        "schema": "DATA",
        "warehouse": "SUPPORT_INTELLIGENCE_WH",
    }


def _get_account_identifier() -> str:
    """Derive the Snowflake account identifier for REST API calls."""
    if os.environ.get("SNOWFLAKE_ACCOUNT"):
        return os.environ["SNOWFLAKE_ACCOUNT"]
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(os.path.expanduser("~/.snowflake/connections.toml"))
        conn_name = SNOWFLAKE_CONNECTION_NAME or "default"
        return config.get(conn_name, "account").strip('"')
    except Exception:
        raise RuntimeError(
            "Cannot determine account. Set SNOWFLAKE_ACCOUNT env var "
            "or ensure connection_name is configured in ~/.snowflake/connections.toml"
        )


def _get_pat() -> str:
    """Get PAT token from env or from connection password."""
    if os.environ.get("SNOWFLAKE_PAT"):
        return os.environ["SNOWFLAKE_PAT"]
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(os.path.expanduser("~/.snowflake/connections.toml"))
        conn_name = SNOWFLAKE_CONNECTION_NAME or "default"
        return config.get(conn_name, "password").strip('"')
    except Exception:
        raise RuntimeError(
            "Cannot determine PAT. Set SNOWFLAKE_PAT env var "
            "or ensure password is configured in ~/.snowflake/connections.toml"
        )


SNOWFLAKE_PARAMS = _get_connection_params()
SNOWFLAKE_PAT = _get_pat()
SNOWFLAKE_ACCOUNT_URL = f"https://{_get_account_identifier()}.snowflakecomputing.com"

SEMANTIC_MODEL_FILE = "@SUPPORT_INTELLIGENCE.DATA.MODELS/semantic_model.yaml"
CORTEX_SEARCH_SERVICE = "SUPPORT_INTELLIGENCE.DATA.KB_SEARCH"


def get_snowpark_session() -> Session:
    return Session.builder.configs(SNOWFLAKE_PARAMS).create()
