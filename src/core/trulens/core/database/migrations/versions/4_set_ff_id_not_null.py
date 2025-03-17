"""Set feedback function id to not nullable.

Revision ID: 4
Revises: 3
Create Date: 2024-08-16 12:44:05.560492
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "4"
down_revision = "3"
branch_labels = None
depends_on = "1"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with op.batch_alter_table(prefix + "feedbacks") as batch_op:
        batch_op.alter_column(
            "feedback_definition_id",
            existing_type=sa.VARCHAR(length=1024),
            nullable=False,
        )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.prefix")

    # ### begin Alembic commands ###
    with op.batch_alter_table(prefix + "feedbacks") as batch_op:
        batch_op.alter_column(
            "feedback_definition_id",
            existing_type=sa.VARCHAR(length=1024),
            nullable=True,
        )
    # ### end Alembic commands ###
