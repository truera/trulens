"""Populate app name and version fields.

Revision ID: 6
Revises: 5
Create Date: 2024-08-16 12:46:49.510690
"""

from alembic import op
from sqlalchemy.orm.session import Session
from trulens.core.database.orm import make_orm_for_prefix
from trulens.core.schema import app as app_schema

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# revision identifiers, used by Alembic.
revision = "6"
down_revision = "5"
branch_labels = None
depends_on = "5"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with Session(bind=op.get_bind()) as session:
        orm = make_orm_for_prefix(table_prefix=prefix)
        apps = session.query(orm.AppDefinition).all()
        if tqdm is not None:
            apps = tqdm(
                apps, desc="Updating app_name and app_version in apps table"
            )
        for app in apps:
            if app.app_name is None:
                app.app_name = "default_app"
            if app.app_version is None:
                app.app_version = app.app_id
            app.app_id = app_schema.AppDefinition._compute_app_id(
                app.app_name, app.app_version
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
            app.app_id = app.app_version
        session.commit()
    # ### end Alembic commands ###
