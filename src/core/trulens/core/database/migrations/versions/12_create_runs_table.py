"""create runs table

Revision ID: 12
Revises: 11
Create Date: 2025-06-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "12"
down_revision = "11"
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.create_table(
        prefix + "runs",
        sa.Column("run_name", sa.VARCHAR(length=256), nullable=False),
        sa.Column("object_name", sa.VARCHAR(length=256), nullable=False),
        sa.Column("object_type", sa.VARCHAR(length=128), nullable=False),
        sa.Column("object_version", sa.VARCHAR(length=128), nullable=True),
        sa.Column("run_status", sa.VARCHAR(length=64), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("run_metadata_json", sa.Text(), nullable=False),
        sa.Column("source_info_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("run_name"),
    )


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_table(prefix + "runs")
