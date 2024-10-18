# ❄️ Logging in Snowflake

o

Snowflake’s fully managed [data warehouse](https://www.snowflake.com/en/data-cloud/workloads/data-warehouse/?utm_cta=website-homepage-workload-card-data-warehouse) provides automatic provisioning, availability, tuning, data protection and more—across clouds and regions—for an unlimited number of users and jobs.

TruLens can write and read from a Snowflake database using a SQLAlchemy connection. This allows you to read, write, persist and share _TruLens_ logs in a _Snowflake_ database.

Here is a guide to logging in _Snowflake_.

## Install the TruLens Snowflake Connector

!!! example "Install using pip"

    ```bash
    pip install trulens-connectors-snowflake
    ```

## Connect TruLens to the Snowflake database

Connecting TruLens to a Snowflake database for logging traces and evaluations only requires passing in an existing [Snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.Session#snowflake.snowpark.Session) or Snowflake [connection parameters](https://docs.snowflake.com/developer-guide/python-connector/python-connector-api#connect).

!!! example "Connect TruLens to your Snowflake database via Snowpark Session"

    ```python
    from snowflake.snowpark import Session
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession
    connection_parameters = {
        account: "<account>",
        user: "<user>",
        password: "<password>",
        database: "<database>",
        schema: "<schema>",
        warehouse: "<warehouse>",
        role: "<role>",
    }
    # Here we create a new Snowpark session, but if we already have one we can use that instead.
    snowpark_session = Session.builder.configs(connection_parameters).create()
    conn = SnowflakeConnector(
        snowpark_session=snowpark_session
    )
    session = TruSession(connector=conn)
    ```

!!! example "Connect TruLens to your Snowflake database via connection parameters"

    ```python
    from trulens.core import TruSession
    from trulens.connectors.snowflake import SnowflakeConnector
    conn = SnowflakeConnector(
        account="<account>",
        user="<user>",
        password="<password>",
        database="<database>",
        schema="<schema>",
        warehouse="<warehouse>",
        role="<role>",
    )
    session = TruSession(connector=conn)
    ```

Once you've instantiated the `TruSession` object with your Snowflake connection, all _TruLens_ traces and evaluations will logged to Snowflake.

## Connect TruLens to the Snowflake database using an engine

In some cases such as when using [key-pair authentication](https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#key-pair-authentication-support), the SQL-alchemy URL does not support the credentials required. In this case, you can instead create and pass a database engine.

When the database engine is created, the private key is then passed through the `connection_args`.

!!! example "Connect TruLens to Snowflake with a database engine"

    ```python
    from trulens.core import Tru
    from sqlalchemy import create_engine
    from snowflake.sqlalchemy import URL
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    load_dotenv()

    with open("rsa_key.p8", "rb") as key:
        p_key= serialization.load_pem_private_key(
            key.read(),
            password=None,
            backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())

    engine = create_engine(URL(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    user=os.environ["SNOWFLAKE_USER"],),
    connect_args={
            'private_key': pkb,
            },
    )

    from trulens.core import TruSession

    session = TruSession(
        database_engine = engine
    )
    ```
