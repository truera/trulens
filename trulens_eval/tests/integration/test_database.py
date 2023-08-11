from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Union
from unittest import main
from unittest import TestCase

from sqlalchemy import Engine
from sqlalchemy import text

from trulens_eval import Feedback
from trulens_eval import FeedbackMode
from trulens_eval import Provider
from trulens_eval import Select
from trulens_eval import Tru
from trulens_eval import TruBasicApp
from trulens_eval.database import orm
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import downgrade_db
from trulens_eval.database.migrations import get_revision_history
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.sqlalchemy_db import AppsExtractor
from trulens_eval.database.sqlalchemy_db import SqlAlchemyDB
from trulens_eval.database.utils import is_legacy_sqlite
from trulens_eval.db import DB
from trulens_eval.db import LocalSQLite


class TestDbV2Migration(TestCase):

    def test_db_migration_sqlite_file(self):
        with clean_db("sqlite_file") as db:
            _test_db_migration(db)

    def test_db_migration_postgres(self):
        with clean_db("postgres") as db:
            _test_db_migration(db)

    def test_db_migration_mysql(self):
        with clean_db("mysql") as db:
            _test_db_migration(db)

    def test_db_consistency_sqlite_file(self):
        with clean_db("sqlite_file") as db:
            _test_db_consistency(db)

    def test_db_consistency_postgres(self):
        with clean_db("postgres") as db:
            _test_db_consistency(db)

    def test_db_consistency_mysql(self):
        with clean_db("mysql") as db:
            _test_db_consistency(db)

    def test_migrate_legacy_sqlite_file(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp).joinpath("legacy.sqlite")

            # populate the database with some legacy data
            legacy_db = LocalSQLite(filename=file)
            fb, app, rec = _populate_data(legacy_db)

            # run migration
            db = SqlAlchemyDB.from_db_url(f"sqlite:///{file}")
            assert is_legacy_sqlite(db.engine)
            db.migrate_database()

            # validate final state
            assert not is_legacy_sqlite(db.engine)
            assert DbRevisions.load(db.engine).in_sync

            # check that database is usable and no data was lost
            assert db.get_app(app.app_id) == json.loads(app.json())
            df_recs, fb_cols = db.get_records_and_feedback([app.app_id])
            assert set(df_recs.columns).issuperset(set(AppsExtractor.app_cols))
            assert df_recs["record_json"][0] == rec.json()
            assert list(fb_cols) == [fb.name]
            df_fb = db.get_feedback(record_id=rec.record_id)
            assert df_fb["type"][0] == app.root_class
            df_defs = db.get_feedback_defs(
                feedback_definition_id=fb.feedback_definition_id
            )
            assert df_defs["feedback_json"][0] == json.loads(fb.json())


class MockFeedback(Provider):

    def length(self, text: str) -> float:  # noqa
        return float(len(text))


@contextmanager
def clean_db(alias: str) -> SqlAlchemyDB:
    with TemporaryDirectory() as tmp:
        url = {
            "sqlite_file":
                f"sqlite:///{Path(tmp).joinpath('test.sqlite')}",
            "postgres":
                "postgresql+psycopg2://pg-test-user:pg-test-pswd@localhost/pg-test-db",
            "mysql":
                "mysql+pymysql://mysql-test-user:mysql-test-pswd@localhost/mysql-test-db",
        }[alias]

        db = SqlAlchemyDB.from_db_url(url)

        downgrade_db(
            db.engine, revision="base"
        )  # drops all tables except `alembic_version`

        with db.engine.connect() as conn:
            conn.execute(text("DROP TABLE alembic_version"))

        yield db


def assert_revision(
    engine: Engine, expected: Union[None, str], status: Literal["in_sync",
                                                                "behind"]
):
    revisions = DbRevisions.load(engine)
    assert revisions.current == expected, f"{revisions.current} != {expected}"
    assert getattr(revisions, status)


def _test_db_migration(db: SqlAlchemyDB):
    engine = db.engine
    history = get_revision_history(engine)
    curr_rev = None

    # apply each upgrade at a time up to head revision
    for i, next_rev in enumerate(history):
        assert int(
            next_rev
        ) == i + 1, f"Versions must be monotonically increasing from 1: {history}"
        assert_revision(engine, curr_rev, "behind")
        upgrade_db(engine, revision=next_rev)
        curr_rev = next_rev

    # validate final state
    assert_revision(engine, history[-1], "in_sync")

    # apply all downgrades
    downgrade_db(engine, revision="base")
    assert_revision(engine, None, "behind")


def _test_db_consistency(db: SqlAlchemyDB):
    db.migrate_database()  # ensure latest revision

    _populate_data(db)
    with db.Session.begin() as session:
        session.delete(
            session.query(orm.AppDefinition).one()
        )  # delete the only app
        assert session.query(orm.Record
                            ).all() == []  # records are deleted in cascade
        assert session.query(orm.FeedbackResult).all() == [
        ]  # feedbacks results are deleted in cascade
        session.query(orm.FeedbackDefinition
                     ).one()  # feedback defs are preserved

    _populate_data(db)
    with db.Session.begin() as session:
        session.delete(
            session.query(orm.Record).one()
        )  # delete the only record
        assert session.query(orm.FeedbackResult).all() == [
        ]  # feedbacks results are deleted in cascade
        session.query(orm.AppDefinition).one()  # apps are preserved
        session.query(orm.FeedbackDefinition
                     ).one()  # feedback defs are preserved


def _populate_data(db: DB):
    tru = Tru()
    tru.db = db  # because of the singleton behavior, db must be changed manually
    fb = Feedback(
        imp=MockFeedback().length,
        feedback_definition_id="mock",
        selectors={"text": Select.RecordOutput},
    )
    app = TruBasicApp(
        text_to_text=lambda x: x,
        app_id="test",
        db=db,
        feedbacks=[fb],
        feedback_mode=FeedbackMode.WITH_APP,
    )
    _, rec = app.call_with_record("boo")
    return fb, app, rec


if __name__ == '__main__':
    main()
