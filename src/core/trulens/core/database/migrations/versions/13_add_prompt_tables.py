"""add prompt management tables

Revision ID: 13
Revises: 12
Create Date: 2026-08-21 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "13"
down_revision = "12"
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.create_table(
        prefix + "prompts",
        sa.Column("prompt_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("slug", sa.VARCHAR(length=256), nullable=False),
        sa.Column("prompt_type", sa.Text(), nullable=False),
        sa.Column("prompt_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("prompt_id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        prefix + "prompt_versions",
        sa.Column("version_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("prompt_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("parent_version_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("content_hash", sa.VARCHAR(length=256), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("prompt_version_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_table(
        prefix + "prompt_labels",
        sa.Column("prompt_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("label", sa.VARCHAR(length=256), nullable=False),
        sa.Column("version_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("prompt_id", "label"),
    )
    op.create_table(
        prefix + "prompt_label_history",
        sa.Column("history_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("prompt_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("label", sa.VARCHAR(length=256), nullable=False),
        sa.Column("previous_version_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("new_version_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("moved_by", sa.VARCHAR(length=256), nullable=True),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("history_id"),
    )


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_table(prefix + "prompt_label_history")
    op.drop_table(prefix + "prompt_labels")
    op.drop_table(prefix + "prompt_versions")
    op.drop_table(prefix + "prompts")
