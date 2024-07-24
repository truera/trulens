# ❄️ Logging in Snowflake

Snowflake’s fully managed [data warehouse](https://www.snowflake.com/en/data-cloud/workloads/data-warehouse/?utm_cta=website-homepage-workload-card-data-warehouse) provides automatic provisioning, availability, tuning, data protection and more—across clouds and regions—for an unlimited number of users and jobs.

TruLens can write and read from a Snowflake database using a SQLAlchemy connection. This allows you to read, write, persist and share _TruLens_ logs in a _Snowflake_ database.

Here is a guide to logging in _Snowflake_.

## Install the [Snowflake SQLAlchemy toolkit](https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy) with the Python Connector


!!! note

    Only snowflake-sqlalchemy version 1.6.1 or greater is supported.

Snowflake SQLAlchemy can be installed from [PyPI](https://pypi.org/project/snowflake-sqlalchemy/):

!!! example "Install Snowflake-SQLAlchemy"

    ```bash
    pip install snowflake-sqlalchemy>=1.6.1
    ```

## Connect TruLens to the Snowflake database

Connecting TruLens to a Snowflake database for logging traces and evaluations only requires passing in a snowflake database URL. Below we've included an example where you can fill in your configuration details to have them formatted into a proper database url and connect to TruLens.

!!! example "Connect TruLens to the Snowflake database"

    ```python
    from trulens.core import Tru
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

Once you've instantiated the `Tru` object with your Snowflake database url, all _TruLens_ traces and evaluations will logged to Snowflake.
