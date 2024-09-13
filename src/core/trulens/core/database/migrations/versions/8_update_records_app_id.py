"""Update app_id column in records table with matching app_id from apps table.

Revision ID: 8
Revises: 7
Create Date: 2024-08-16 12:46:49.510690
"""

from alembic import op
from sqlalchemy.orm.session import Session
from trulens.core.database.orm import make_orm_for_prefix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# revision identifiers, used by Alembic.
revision = "8"
down_revision = "7"
branch_labels = None
depends_on = "6"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with Session(bind=op.get_bind()) as session:
        orm = make_orm_for_prefix(table_prefix=prefix)

        apps = session.query(orm.AppDefinition).all()
        if tqdm is not None:
            apps = tqdm(apps, desc="Updating app_id in records table")
        for app in apps:
            op.execute(
                f"UPDATE {prefix + 'records'} SET app_id = '{app.app_id}' WHERE app_id = '{app.app_version}'"
            )

        session.commit()
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###

    with Session(bind=op.get_bind()) as session:
        orm = make_orm_for_prefix(table_prefix=prefix)

        apps = session.query(orm.AppDefinition).all()
        for app in apps:
            op.execute(
                f"UPDATE {prefix + 'records'} SET app_id = '{app.app_version}' WHERE app_id = '{app.app_id}'"
            )
        session.commit()
    # ### end Alembic commands ###
