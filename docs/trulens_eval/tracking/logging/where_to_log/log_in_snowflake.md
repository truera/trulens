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

## Connect TruLens to the Snowflake database

!!! example "Connect TruLens to the Snowflake database"

    ```python
    from trulens_eval import Tru
    tru = Tru(database_url=(
        'snowflake://{user}:{password}@{account_identifier}/'
        '{database}/{schema}?warehouse={warehouse}&role={role}'
    ).format(
        user='<user>',
        password='<password>',
        account_identifier='<account-identifer>',
        database='<database>',
        schema='<schema>',
        warehouse='<warehouse>',
        role='<role>'
    ))
    ```
