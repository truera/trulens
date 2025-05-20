"""Make app name and app version fields not nullable.

Revision ID: 7
Revises: 6
Create Date: 2024-08-27 11:09:26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7"
down_revision = "6"
branch_labels = None
depends_on = "6"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with op.batch_alter_table(prefix + "apps") as batch_op:
        batch_op.alter_column(
            "app_name",
            existing_type=sa.VARCHAR(length=1024),
            nullable=False,
        )
        batch_op.alter_column(
            "app_version",
            existing_type=sa.VARCHAR(length=1024),
            nullable=False,
        )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with op.batch_alter_table(prefix + "apps") as batch_op:
        batch_op.alter_column(
            "app_name",
            existing_type=sa.VARCHAR(length=1024),
            nullable=True,
        )
        batch_op.alter_column(
            "app_version",
            existing_type=sa.VARCHAR(length=1024),
            nullable=True,
        )
    # ### end Alembic commands ###
