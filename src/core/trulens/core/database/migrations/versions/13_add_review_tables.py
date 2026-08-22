"""Add human review queue tables.

Revision ID: 13
Revises: 12
Create Date: 2026-08-22 00:00:00.000000
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
        prefix + "review_queues",
        sa.Column("review_queue_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("name", sa.VARCHAR(length=256), nullable=False),
        sa.Column("review_queue_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("review_queue_id"),
    )
    op.create_table(
        prefix + "review_items",
        sa.Column("review_item_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("review_queue_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("target_type", sa.VARCHAR(length=64), nullable=False),
        sa.Column("target_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("state", sa.VARCHAR(length=32), nullable=False),
        sa.Column("claim_token", sa.VARCHAR(length=256), nullable=True),
        sa.Column("claimed_at", sa.Float(), nullable=True),
        sa.Column("claimed_by", sa.VARCHAR(length=256), nullable=True),
        sa.Column("current_review_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("review_item_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("review_item_id"),
        sa.ForeignKeyConstraint(
            ["review_queue_id"],
            [prefix + "review_queues.review_queue_id"],
        ),
    )
    op.create_index(
        f"ix_{prefix}review_items_queue_state",
        prefix + "review_items",
        ["review_queue_id", "state"],
    )
    op.create_table(
        prefix + "human_reviews",
        sa.Column("human_review_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("target_type", sa.VARCHAR(length=64), nullable=False),
        sa.Column("target_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("verdict", sa.VARCHAR(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("reviewer", sa.VARCHAR(length=256), nullable=True),
        sa.Column("review_queue_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("supersedes_id", sa.VARCHAR(length=256), nullable=True),
        sa.Column("human_review_json", sa.Text(), nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("human_review_id"),
    )
    op.create_index(
        f"ix_{prefix}human_reviews_target",
        prefix + "human_reviews",
        ["target_type", "target_id"],
    )


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    op.drop_index(
        f"ix_{prefix}human_reviews_target", table_name=prefix + "human_reviews"
    )
    op.drop_table(prefix + "human_reviews")
    op.drop_index(
        f"ix_{prefix}review_items_queue_state",
        table_name=prefix + "review_items",
    )
    op.drop_table(prefix + "review_items")
    op.drop_table(prefix + "review_queues")
