# Where to Log

By default, all data is logged to the current working directory to `default.sqlite` (`sqlite:///default.sqlite`).

## Connecting with a Database URL

Data can be logged to a SQLAlchemy-compatible referred to by `database_url` in the format `dialect+driver://username:password@host:port/database`.

See [this article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for more details on SQLAlchemy database URLs.

For example, for Postgres database `trulens` running on `localhost` with username `trulensuser` and password `password` set up a connection like so.

!!! example "Connecting with a Database URL"

    ```python
    from trulens.core import TruSession
    tru = TruSession(database_url="postgresql://trulensuser:password@localhost/trulens")
    ```

After which you should receive the following message:

```
🦑 Tru initialized with db url postgresql://trulensuser:password@localhost/trulens.
```

## Connecting to a Database Engine

Data can also logged to a SQLAlchemy-compatible engine referred to by `database_engine`. This is useful when you need to pass keyword args in addition to the database URL to connect to your database, such as [`connect_args`](https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine.params.connect_args).

See [this article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for more details on SQLAlchemy database engines.

!!! example "Connecting with a Database Engine"

    ```python
    from trulens.core import Tru
    from sqlalchemy import create_engine

    engine = create_engine(
        "postgresql://trulensuser:password@localhost/trulens",
        connect_args={"connection_factory": MyConnectionFactory},
    )

    tru = Tru(database_engine=engine)
    ```

After which you should receive the following message:

```
🦑 Tru initialized with db url postgresql://trulensuser:password@localhost/trulens.
