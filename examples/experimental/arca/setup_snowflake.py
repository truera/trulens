"""
Setup Snowflake objects for ARCA

Uses externalbrowser authentication (opens browser for SSO login).
Assumes existing database, schema, and warehouse.
Creates:
- Stage
- Example UDF
"""

import logging
import os

from dotenv import load_dotenv
from snowflake.snowpark import Session

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_snowpark_session() -> Session:
    """Create Snowpark session using externalbrowser authentication."""
    config = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "authenticator": "externalbrowser",
    }

    return Session.builder.configs(config).create()


def setup_objects(session: Session):
    """Create required Snowflake objects (assumes DB, schema, warehouse exist)."""

    # Get names from environment variables
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    # Use existing database, schema, and warehouse
    logger.info(f"Using database: {database}")
    session.sql(f"USE DATABASE {database}").collect()

    logger.info(f"Using schema: {schema}")
    session.sql(f"USE SCHEMA {schema}").collect()

    logger.info(f"Using warehouse: {warehouse}")
    session.sql(f"USE WAREHOUSE {warehouse}").collect()

    # Create stage
    # SQL: CREATE STAGE IF NOT EXISTS ARCA_STAGE;
    logger.info("Creating stage ARCA_STAGE...")
    session.sql("CREATE STAGE IF NOT EXISTS ARCA_STAGE").collect()

    # Create example UDF for testing
    # SQL: CREATE OR REPLACE FUNCTION calculate_revenue(region VARCHAR, start_date DATE, end_date DATE)
    #      RETURNS FLOAT
    #      LANGUAGE PYTHON
    #      RUNTIME_VERSION='3.10'
    #      HANDLER='calculate_revenue'
    logger.info("Creating example UDF...")
    session.sql("""
        CREATE OR REPLACE FUNCTION calculate_revenue(region VARCHAR, start_date DATE, end_date DATE)
        RETURNS FLOAT
        LANGUAGE PYTHON
        RUNTIME_VERSION='3.10'
        HANDLER='calculate_revenue'
        AS $$
def calculate_revenue(region, start_date, end_date):
    # Mock implementation - returns fake revenue
    import random
    random.seed(hash(region + str(start_date) + str(end_date)))
    return random.uniform(1000000, 10000000)
$$
    """).collect()

    logger.info("Setup complete!")
    logger.info(f"Database: {database}")
    logger.info(f"Schema: {schema}")
    logger.info(f"Warehouse: {warehouse}")
    logger.info("Stage: ARCA_STAGE")
    logger.info("UDF: calculate_revenue(region, start_date, end_date)")


def main():
    """Main setup function."""
    # Check environment variables
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.error("Please set:")
        logger.error("  SNOWFLAKE_ACCOUNT=your-account")
        logger.error("  SNOWFLAKE_USER=your-username")
        return 1

    # Create session
    logger.info("Connecting to Snowflake...")
    session = get_snowpark_session()

    # Setup objects
    setup_objects(session)

    # Close session
    session.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
