"""add indexes to event table

Revision ID: 11
Revises: 10
Create Date: 2025-04-15 00:00:00.000000
"""

from alembic import op
from trulens.core.otel.utils import is_otel_tracing_enabled

revision = "11"
down_revision = "10"
branch_labels = None
depends_on = None


def _use_event_table():
    return is_otel_tracing_enabled()


def upgrade(config) -> None:
    if not _use_event_table():
        return

    prefix = config.get_main_option("trulens.table_prefix")
    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    table_name = prefix + "events"

    op.create_index(
        f"ix_{table_name}_start_timestamp",
        table_name,
        ["start_timestamp"],
    )
    op.create_index(
        f"ix_{table_name}_timestamp",
        table_name,
        ["timestamp"],
    )


def downgrade(config) -> None:
    if not _use_event_table():
        return

    prefix = config.get_main_option("trulens.table_prefix")
    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    table_name = prefix + "events"

    op.drop_index(f"ix_{table_name}_timestamp", table_name)
    op.drop_index(f"ix_{table_name}_start_timestamp", table_name)
