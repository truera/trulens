"""Add app name and version fields.

Revision ID: 5
Revises: 4
Create Date: 2024-08-16 12:46:49.510690
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm.session import Session

# revision identifiers, used by Alembic.
revision = "5"
down_revision = "4"
branch_labels = None
depends_on = "1"


def _is_snowflake_engine(config) -> bool:
    # This is a bit of a hack to guess if we're dealing with a Snowflake engine as they can't handle renaming columns correctly.
    return config.attributes["engine"].name == "snowflake"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    if _is_snowflake_engine(config):
        with Session(bind=op.get_bind()) as session:
            session.execute(
                sa.text(
                    f"ALTER TABLE {prefix}apps RENAME COLUMN app_id TO app_name"
                )
            )
    else:
        op.alter_column(
            prefix + "apps",
            "app_id",
            new_column_name="app_name",
        )
    op.add_column(
        prefix + "apps",
        sa.Column(
            "app_id", sa.VARCHAR(length=256), nullable=False, primary_key=True
        ),
    )
    op.add_column(
        prefix + "apps",
        sa.Column("app_version", sa.VARCHAR(length=256), nullable=False),
    )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.drop_column(prefix + "apps", "app_version")
    op.drop_column(prefix + "apps", "app_id")
    if _is_snowflake_engine(config):
        with Session(bind=op.get_bind()) as session:
            session.execute(
                sa.text(
                    f"ALTER TABLE {prefix}apps RENAME COLUMN app_name TO app_id"
                )
            )
    else:
        op.alter_column(
            prefix + "apps",
            "app_name",
            new_column_name="app_id",
        )
    # ### end Alembic commands ###
