"""Add dataset version tables.

Adds immutable, content-addressed dataset snapshots alongside the existing
mutable `dataset` / `ground_truth` tables. Existing rows are left untouched;
they are exposed as version zero by the compatibility loader in
`trulens.core.database.sqlalchemy`.

Revision ID: 13
Revises: 12
Create Date: 2026-08-21 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "13"
down_revision = "12"
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.create_table(
        prefix + "dataset_version",
        sa.Column("dataset_version_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("dataset_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column(
            "parent_dataset_version_id",
            sa.VARCHAR(length=256),
            nullable=True,
        ),
        sa.Column("version_index", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.VARCHAR(length=256), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("dataset_version_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"], [prefix + "dataset.dataset_id"]
        ),
        sa.PrimaryKeyConstraint("dataset_version_id"),
        sa.UniqueConstraint(
            "dataset_id",
            "version_index",
            name=f"uq_{prefix}dataset_version_dataset_version_index",
        ),
    )
    op.create_table(
        prefix + "dataset_version_item",
        sa.Column("dataset_version_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("item_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("input_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("dataset_version_item_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            [prefix + "dataset_version.dataset_version_id"],
        ),
        sa.PrimaryKeyConstraint("dataset_version_id", "item_id"),
    )


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_table(prefix + "dataset_version_item")
    op.drop_table(prefix + "dataset_version")
