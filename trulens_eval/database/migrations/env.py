from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from trulens_eval.database import base as mod_db
from trulens_eval.database.orm import make_orm_for_prefix

# Gives access to the values within the alembic.ini file
config = context.config

# Run this block only if Alembic was called from the command-line
#if config.get_main_option("calling_context", default="CLI") == "CLI":
# NOTE(piotrm): making this run always so users can configure alembic.ini as
# they see fit.

# Interpret the `alembic.ini` file for Python logging.
if config.config_file_name is not None:
    if not os.path.exists(config.config_file_name):
        raise FileNotFoundError(
            f"Alembic config file not found: {config.config_file_name}."
        )

    fileConfig(config.config_file_name)

# Get `sqlalchemy.url` from the environment.
if config.get_main_option("sqlalchemy.url", None) is None:
    config.set_main_option(
        "sqlalchemy.url", os.environ.get("SQLALCHEMY_URL", "")
    )

# Get `trulens.table_prefix` from the environment.
prefix = config.get_main_option(
    "trulens.table_prefix"
) or mod_db.DEFAULT_DATABASE_PREFIX

orm = make_orm_for_prefix(table_prefix=prefix)

# Database schema information
target_metadata = orm.metadata

url = config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=prefix + "alembic_version"
    )

    with context.begin_transaction():
        context.run_migrations(confi=config)


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    if not (engine := config.attributes.get("engine")):
        engine = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=prefix + "alembic_version"
        )

        with context.begin_transaction():
            context.run_migrations(config=config)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
