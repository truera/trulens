"""Add run_location column to feedback_defs table.

Revision ID: 2
Revises: 1
Create Date: 2024-08-15 15:43:22.560492
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2"
down_revision = "1"
branch_labels = None
depends_on = "1"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.add_column(
        prefix + "feedback_defs",
        sa.Column("run_location", sa.Text(), nullable=True),
    )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.drop_column(prefix + "feedback_defs", "run_location")
    # ### end Alembic commands ###
