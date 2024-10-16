"""Add groundtruth and dataset tables.

Revision ID: 3
Revises: 2
Create Date: 2024-08-17 15:33:09.416935
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "3"
down_revision = "2"
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.create_table(
        prefix + "dataset",
        sa.Column("dataset_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("dataset_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id"),
    )
    op.create_table(
        prefix + "ground_truth",
        sa.Column("ground_truth_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("ground_truth_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("ground_truth_id"),
    )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.drop_table(prefix + "ground_truth")
    op.drop_table(prefix + "dataset")
    # ### end Alembic commands ###
