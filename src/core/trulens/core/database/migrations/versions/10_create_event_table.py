"""create event table

Revision ID: 10
Revises: 9
Create Date: 2024-12-11 09:32:48.976169
"""

from alembic import op
import sqlalchemy as sa
from trulens.core.otel.utils import is_otel_tracing_enabled

# revision identifiers, used by Alembic.
revision = "10"
down_revision = "9"
branch_labels = None
depends_on = None


def _use_event_table():
    # We only use event table if specifically enabled as it requires the often
    # unsupported JSON type and is for temporary testing purposes anyway.
    return is_otel_tracing_enabled()


def upgrade(config) -> None:
    if not _use_event_table():
        return

    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # Check if the database is Snowflake
    dict_type = sa.JSON()
    is_snowflake = config.get_main_option("sqlalchemy.url", "").startswith(
        "snowflake"
    )
    if is_snowflake:
        # Snowflake does not support JSON type, so we use TEXT instead
        # for compatibility with the rest of the code.
        dict_type = sa.TEXT()

    op.create_table(
        prefix + "events",
        sa.Column("event_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("record", dict_type, nullable=False),
        sa.Column("record_attributes", dict_type, nullable=False),
        sa.Column("record_type", sa.VARCHAR(length=256), nullable=False),
        sa.Column("resource_attributes", dict_type, nullable=False),
        sa.Column("start_timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("trace", dict_type, nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
    )


def downgrade(config) -> None:
    if not _use_event_table():
        return

    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_table(prefix + "events")
