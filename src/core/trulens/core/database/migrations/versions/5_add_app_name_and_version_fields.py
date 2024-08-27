"""Add app name and version fields.

Revision ID: 5
Revises: 4
Create Date: 2024-08-16 12:46:49.510690
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "5"
down_revision = "4"
branch_labels = None
depends_on = "1"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.add_column(
        prefix + "apps",
        sa.Column(
            "app_name",
            sa.VARCHAR(length=256),
            default="default_app",
        ),
    )
    op.add_column(
        prefix + "apps",
        sa.Column(
            "app_version",
            sa.VARCHAR(length=256),
            default="base",
        ),
    )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.drop_column(prefix + "apps", "app_name")
    op.drop_column(prefix + "apps", "app_version")
    # ### end Alembic commands ###
