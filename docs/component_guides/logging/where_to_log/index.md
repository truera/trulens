# Where to Log

By default, all data is logged to the current working directory to `default.sqlite` (`sqlite:///default.sqlite`).

## Supported Databases

TruLens supports logging to any SQLAlchemy-compatible database:

- **SQLite** (default) - No additional setup required
- **[PostgreSQL](log_in_postgres.md)** - Open-source relational database
- **[Snowflake](log_in_snowflake.md)** - Cloud data warehouse
- **MySQL** - Open-source relational database

## Connecting with a Database URL

Data can be logged to a SQLAlchemy-compatible database referred to by `database_url` in the format `dialect+driver://username:password@host:port/database`.

See [this article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for more details on SQLAlchemy database URLs.

!!! example "Connecting with a Database URL"

    ```python
    from trulens.core import TruSession

    session = TruSession(
        database_url="postgresql://username:password@localhost:5432/mydatabase"
    )
    ```

## Connecting with a Database Engine

Data can also be logged to a SQLAlchemy-compatible engine referred to by `database_engine`. This is useful when you need to pass keyword args in addition to the database URL to connect to your database, such as [`connect_args`](https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine.params.connect_args).

!!! example "Connecting with a Database Engine"

    ```python
    from trulens.core import TruSession
    from sqlalchemy import create_engine

    engine = create_engine(
        "postgresql://username:password@localhost:5432/mydatabase",
        connect_args={"sslmode": "require"},
    )

    session = TruSession(database_engine=engine)
    ```
