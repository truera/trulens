"""create event table

Revision ID: 10
Revises: 9
Create Date: 2024-12-11 09:32:48.976169
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "10"
down_revision = "9"
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.create_table(
        prefix + "events",
        sa.Column("event_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
        sa.Column("record_attributes", sa.JSON(), nullable=False),
        sa.Column("record_type", sa.VARCHAR(length=256), nullable=False),
        sa.Column("resource_attributes", sa.JSON(), nullable=False),
        sa.Column("start_timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("trace", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
    )


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_table(prefix + "events")
