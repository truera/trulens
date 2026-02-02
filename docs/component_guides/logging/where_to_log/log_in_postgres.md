# ![PostgreSQL](../../../assets/images/logos/postgresql.svg){ width="30" } Logging in PostgreSQL

[PostgreSQL](https://www.postgresql.org/) is a powerful, open-source relational database system with a strong reputation for reliability, feature robustness, and performance.

TruLens can write and read from a PostgreSQL database using SQLAlchemy. This allows you to read, write, persist and share TruLens logs in a PostgreSQL database.

Here is a guide to logging in PostgreSQL.

## Prerequisites

### Install PostgreSQL Driver

TruLens requires a **synchronous** PostgreSQL driver. Install one of the following:

!!! example "Install using pip"

    ```bash
    # Recommended: psycopg2-binary (includes compiled binaries)
    pip install psycopg2-binary

    # Alternative: psycopg2 (requires PostgreSQL dev libraries)
    pip install psycopg2

    # Alternative: psycopg v3
    pip install psycopg
    ```

!!! warning "Do NOT use asyncpg"

    **`asyncpg` is not supported.** It is an async-only driver and is incompatible with TruLens's synchronous SQLAlchemy operations. Using `asyncpg` will result in a `MissingGreenlet` error.

### Start PostgreSQL (Optional: Docker)

If you don't have a PostgreSQL instance, you can start one using Docker:

```bash
docker run -d \
    --name trulens-postgres \
    -e POSTGRES_DB=trulens \
    -e POSTGRES_USER=trulensuser \
    -e POSTGRES_PASSWORD=password \
    -p 5432:5432 \
    postgres:15-alpine
```

## Connect TruLens to PostgreSQL

### Using a Database URL

The simplest way to connect is with a database URL in the format:
`postgresql://username:password@host:port/database`

!!! example "Connect TruLens to PostgreSQL"

    ```python
    from trulens.core import TruSession

    session = TruSession(
        database_url="postgresql://trulensuser:password@localhost:5432/trulens"
    )
    ```

After which you should see:

```
🦑 Initialized with db url postgresql://trulensuser:***@localhost:5432/trulens.
```

### Using a Database Engine

For more control over the connection (e.g., SSL settings, connection pooling), create a SQLAlchemy engine:

!!! example "Connect TruLens to PostgreSQL with SSL"

    ```python
    from trulens.core import TruSession
    from sqlalchemy import create_engine

    engine = create_engine(
        "postgresql://trulensuser:password@localhost:5432/trulens",
        connect_args={"sslmode": "require"},
    )

    session = TruSession(database_engine=engine)
    ```

### Using Environment Variables (Recommended for Production)

!!! example "Connect using environment variables"

    ```python
    import os
    from trulens.core import TruSession

    session = TruSession(
        database_url=os.environ["DATABASE_URL"],
        database_redact_keys=True,  # Recommended: redact sensitive data
    )
    ```

## Connection URL Formats

TruLens supports various PostgreSQL connection URL formats:

| Format | Example |
|--------|---------|
| Basic | `postgresql://user:pass@host:port/db` |
| With psycopg2 | `postgresql+psycopg2://user:pass@host:port/db` |
| With psycopg v3 | `postgresql+psycopg://user:pass@host:port/db` |
| With SSL | `postgresql://user:pass@host:port/db?sslmode=require` |

## Full Example

For a complete working example, see the [Log in PostgreSQL notebook](../../../../examples/expositional/logging/log_in_postgres.ipynb).
