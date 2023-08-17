from sqlalchemy.sql import text

from trulens_eval import Tru

with Tru().db.engine.connect() as c:
    c.execute(text("update alembic_version set version_num=99999"))
    c.commit()
