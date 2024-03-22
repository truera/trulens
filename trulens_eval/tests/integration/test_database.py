from contextlib import contextmanager
import json
from pathlib import Path
import shutil
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
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import downgrade_db
from trulens_eval.database.migrations import get_revision_history
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.sqlalchemy import AppsExtractor
from trulens_eval.database.sqlalchemy import SQLAlchemyDB
from trulens_eval.database.utils import is_legacy_sqlite
from trulens_eval.database.base import DB
from trulens_eval.database.legacy.sqlite import LocalSQLite


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

    def test_future_db(self):
        # Check handling of database that is newer than the current
        # trulens_eval's db version. We expect a warning and exception.

        for folder in (Path(__file__).parent.parent.parent /
                       "release_dbs").iterdir():
            _dbfile = folder / "default.sqlite"

            if not "infty" in str(folder):
                # Future/unknown dbs have "infty" in their folder name.
                continue

            with self.subTest(msg=f"use future db {folder.name}"):
                with TemporaryDirectory() as tmp:
                    dbfile = Path(tmp) / f"default-{folder.name}.sqlite"
                    shutil.copy(str(_dbfile), str(dbfile))

                    self._test_future_db(dbfile=dbfile)

    def _test_future_db(self, dbfile: Path = None):
        db = SQLAlchemyDB.from_db_url(f"sqlite:///{dbfile}")
        self.assertFalse(is_legacy_sqlite(db.engine))

        # Migration should state there is a future version present which we
        # cannot migrate.
        with self.assertRaises(DatabaseVersionException) as e:
            db.migrate_database()

        self.assertEqual(
            e.exception.reason, DatabaseVersionException.Reason.AHEAD
        )

        # Trying to use it anyway should also produce the exception.
        with self.assertRaises(DatabaseVersionException) as e:
            db.get_records_and_feedback()

        self.assertEqual(
            e.exception.reason, DatabaseVersionException.Reason.AHEAD
        )

    def test_migrate_legacy_legacy_sqlite_file(self):
        # Migration from non-latest lagecy db files all the way to v2 database.
        # This involves migrating the legacy dbs to the latest legacy first.

        for folder in (Path(__file__).parent.parent.parent /
                       "release_dbs").iterdir():
            _dbfile = folder / "default.sqlite"

            if "infty" in str(folder):
                # This is a db marked with version 99999. See the future_db tests
                # for use.
                continue

            with self.subTest(msg=f"migrate from {folder.name} folder"):
                with TemporaryDirectory() as tmp:
                    dbfile = Path(tmp) / f"default-{folder.name}.sqlite"
                    shutil.copy(str(_dbfile), str(dbfile))

                    self._test_migrate_legacy_legacy_sqlite_file(dbfile=dbfile)

    def _test_migrate_legacy_legacy_sqlite_file(self, dbfile: Path = None):
        # run migration
        db = SQLAlchemyDB.from_db_url(f"sqlite:///{dbfile}")
        self.assertTrue(is_legacy_sqlite(db.engine))
        db.migrate_database()

        # validate final state
        self.assertFalse(is_legacy_sqlite(db.engine))
        self.assertTrue(DbRevisions.load(db.engine).in_sync)

        records, feedbacks = db.get_records_and_feedback()

        # Very naive checks:
        self.assertGreater(len(records), 0)
        self.assertGreater(len(feedbacks), 0)

    def test_migrate_legacy_sqlite_file(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp).joinpath("legacy.sqlite")

            # populate the database with some legacy data
            legacy_db = LocalSQLite(filename=file)
            fb, app, rec = _populate_data(legacy_db)

            # run migration
            db = SQLAlchemyDB.from_db_url(f"sqlite:///{file}")
            self.assertTrue(is_legacy_sqlite(db.engine))
            db.migrate_database()

            # validate final state
            self.assertFalse(is_legacy_sqlite(db.engine))
            self.assertTrue(DbRevisions.load(db.engine).in_sync)

            # check that database is usable and no data was lost
            self.assertEqual(
                db.get_app(app.app_id), json.loads(app.model_dump_json())
            )
            df_recs, fb_cols = db.get_records_and_feedback([app.app_id])
            self.assertTrue(
                set(df_recs.columns).issuperset(set(AppsExtractor.app_cols))
            )
            self.assertEqual(df_recs["record_json"][0], rec.model_dump_json())
            self.assertEqual(list(fb_cols), [fb.name])

            df_fb = db.get_feedback(record_id=rec.record_id)

            self.assertEqual(df_fb["type"][0], app.root_class)
            df_defs = db.get_feedback_defs(
                feedback_definition_id=fb.feedback_definition_id
            )
            self.assertEqual(
                df_defs["feedback_json"][0], json.loads(fb.model_dump_json())
            )


class MockFeedback(Provider):

    def length(self, text: str) -> float:  # noqa
        return float(len(text))


@contextmanager
def clean_db(alias: str) -> SQLAlchemyDB:
    with TemporaryDirectory() as tmp:
        url = {
            "sqlite_file":
                f"sqlite:///{Path(tmp).joinpath('test.sqlite')}",
            "postgres":
                "postgresql+psycopg2://pg-test-user:pg-test-pswd@localhost/pg-test-db",
            "mysql":
                "mysql+pymysql://mysql-test-user:mysql-test-pswd@localhost/mysql-test-db",
        }[alias]

        db = SQLAlchemyDB.from_db_url(url)

        downgrade_db(
            db.engine, revision="base"
        )  # drops all tables except `db.version_table`

        with db.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE {db.version_table}"))

        yield db


def assert_revision(
    engine: Engine, expected: Union[None, str], status: Literal["in_sync",
                                                                "behind"]
):
    revisions = DbRevisions.load(engine)
    assert revisions.current == expected, f"{revisions.current} != {expected}"
    assert getattr(revisions, status)


def _test_db_migration(db: SQLAlchemyDB):
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


def _test_db_consistency(db: SQLAlchemyDB):
    db.migrate_database()  # ensure latest revision

    _populate_data(db)
    with db.session.begin() as session:
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
    with db.session.begin() as session:
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
    _, rec = app.with_record(app.app.__call__, "boo")
    return fb, app, rec


if __name__ == '__main__':
    main()
