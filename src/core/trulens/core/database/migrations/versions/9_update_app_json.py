"""Update app_json with app_id from apps table.

Revision ID: 9
Revises: 8
Create Date: 2024-08-16 12:46:49.510690
"""

import json

from alembic import op
from sqlalchemy.orm.session import Session
from trulens.core.database.orm import make_orm_for_prefix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# revision identifiers, used by Alembic.
revision = "9"
down_revision = "8"
branch_labels = None
depends_on = "7"


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    with Session(bind=op.get_bind()) as session:
        orm = make_orm_for_prefix(table_prefix=prefix)

        apps = session.query(orm.AppDefinition).all()
        if tqdm is not None:
            apps = tqdm(apps, desc="Updating app_json in apps table")
        for app in apps:
            if app.app_json is None:
                continue
            app_json = json.loads(app.app_json)
            app_json["app_id"] = app.app_id
            if "app_name" not in app_json:
                app_json["app_name"] = app.app_name
            if "app_version" not in app_json:
                app_json["app_version"] = app.app_version
            app.app_json = json.dumps(app_json)
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
            if app.app_json is None:
                continue
            app_json = json.loads(app.app_json)
            app_json["app_id"] = app.app_version
            if "app_name" in app_json:
                del app_json["app_name"]
            if "app_version" in app_json:
                del app_json["app_version"]
            app.app_json = json.dumps(app_json)
        session.commit()
    # ### end Alembic commands ###
