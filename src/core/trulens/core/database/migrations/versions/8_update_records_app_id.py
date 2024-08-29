"""Populate app name and version fields.

Revision ID: 8
Revises: 7
Create Date: 2024-08-16 12:46:49.510690
"""

from alembic import op
from sqlalchemy.orm.session import Session
from trulens.core.database.orm import make_orm_for_prefix

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
        records = session.query(orm.Record).all()

        apps = session.query(orm.AppDefinition).all()
        app_id_mapping = {app.app_version: app.app_id for app in apps}

        for record in records:
            if record.app_id is not None:
                record.app_id = app_id_mapping[record.app_id]
        session.commit()
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###

    with Session(bind=op.get_bind()) as session:
        orm = make_orm_for_prefix(table_prefix=prefix)

        records = session.query(orm.Record).all()

        apps = session.query(orm.AppDefinition).all()
        app_id_mapping = {app.app_id: app.app_version for app in apps}

        for record in records:
            if record.app_id is not None:
                record.app_id = app_id_mapping[record.app_id]
        session.commit()
    # ### end Alembic commands ###
