# ❄️ Logging in Snowflake

Snowflake’s fully managed [data warehouse](https://www.snowflake.com/en/data-cloud/workloads/data-warehouse/?utm_cta=website-homepage-workload-card-data-warehouse) provides automatic provisioning, availability, tuning, data protection and more—across clouds and regions—for an unlimited number of users and jobs.

TruLens can write and read from a Snowflake database using a SQLAlchemy connection. This allows you to read, write, persist and share _TruLens_ logs in a _Snowflake_ database.

Here is a _working_ guide to logging in _Snowflake_.

## Install the [Snowflake SQLAlchemy toolkit](https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy) with the Python Connector

For now, we need to use a working branch of snowflake-sqlalchemy that supports sqlalchemy 2.0.

!!! example "Install Snowflake-SQLAlchemy"

    ```bash
    # Clone the Snowflake github repo:
    git clone git@github.com:snowflakedb/snowflake-sqlalchemy.git

    # Check out the sqlalchemy branch:
    git checkout SNOW-1058245-sqlalchemy-20-support

    # Install hatch:
    pip install hatch

    # Build snowflake-sqlalchemy via hatch:
    python -m hatch build --clean

    # Install snowflake-sqlalchemy
    pip install dist/*.whl
    ```

## Install TruLens

For now, we need to use a working branch of TruLens that supports snowflake databases.

!!! example "Install TruLens"

    ```bash
    pip uninstall trulens_eval -y # to remove existing PyPI version
    pip install git+https://github.com/truera/trulens@garett/snowflake-test#subdirectory=trulens_eval

    ```

## Create required TruLens tables and schemas in Snowflake.

Navigate to the Snowflake console and create tables with SQL. Then run the following commands to create the tables. This is not required if the tables have already been created.

!!! example "Create TruLens tables in Snowflake database"

    ```
    CREATE TABLE IF NOT EXISTS TRULENS_ALEMBIC_VERSION (
    version_num NUMBER NOT NULL PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS TRULENS_APPS (
    app_id TEXT NOT NULL PRIMARY KEY,
    app_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS TRULENS_FEEDBACK_DEFS (
    feedback_definition_id TEXT NOT NULL PRIMARY KEY,
    feedback_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS TRULENS_FEEDBACKS (
    feedback_result_id TEXT NOT NULL PRIMARY KEY,
    record_id TEXT NOT NULL,
    feedback_definition_id TEXT,
    last_ts FLOAT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    calls_json TEXT NOT NULL,
    result FLOAT,
    name TEXT NOT NULL,
    cost_json TEXT NOT NULL,
    multi_result TEXT
    );

    CREATE TABLE IF NOT EXISTS TRULENS_RECORDS (
    record_id TEXT NOT NULL PRIMARY KEY,
    app_id TEXT NOT NULL,
    input TEXT,
    output TEXT,
    record_json TEXT NOT NULL,
    tags TEXT NOT NULL,
    ts FLOAT NOT NULL,
    cost_json TEXT NOT NULL,
    perf_json TEXT NOT NULL
    );
    ```

## Connect TruLens to the Snowflake database.

!!! example "Connect TruLens to the Snowflake database"

    ```python
    from trulens_eval import Tru
    tru = Tru(database_url=(
        'snowflake://{user}:{password}@{account_identifier}/'
        'TRULENS_TEST_V0/TRULENS?warehouse=COMPUTE_WH&role=ACCOUNTADMIN'
    ).format(
        user='<user>',
        password='<password>',
        account_identifier='<account-identifer>', # oaztwkp-bnb75599
    ))
    ```
