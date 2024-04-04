"""Database Tests

Some of the tests in file require a running docker container which hosts the
tested databases. See `trulens_eval/docker/test-database.yaml` and/or
`trulens_eval/Makefile` target `test-database` for how to get this container
running.

- Tests migration of old databases to new ones.

- Tests uses of various DB vender types. A subset of types supported by
  sqlalchemy:

    - sqlite
    - postgres (in docker)
    - mysql (in docker)

- Tests database options like prefix.

- Tests database utilities:

    - `copy_database`
"""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, Literal, Union
from unittest import main
from unittest import TestCase

import pandas as pd
from sqlalchemy import Engine

from trulens_eval import Feedback
from trulens_eval import FeedbackMode
from trulens_eval import Provider
from trulens_eval import Select
from trulens_eval import Tru
from trulens_eval import TruBasicApp
from trulens_eval.database.base import DB
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.database.legacy.sqlite import LocalSQLite
from trulens_eval.database.migrations import DbRevisions
from trulens_eval.database.migrations import downgrade_db
from trulens_eval.database.migrations import get_revision_history
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.sqlalchemy import AppsExtractor
from trulens_eval.database.sqlalchemy import SQLAlchemyDB
from trulens_eval.database.utils import is_legacy_sqlite


class TestDBSpecifications(TestCase):
    """
    Tests for database options.
    """

    def test_prefix(self):

        db_types = ["sqlite_file"]#, "postgres", "mysql", "sqlite_memory"
        # sqlite_memory might have problems with multithreading of tests

        for db_type in db_types:
            with self.subTest(msg=f"prefix for {db_type}"):
                with clean_db(db_type, table_prefix="test_") as db:

                    _populate_data(db)

                    tables = ["apps", "records", "feedback_defs", "feedbacks"]

                    for table in tables:
                        print(table)
                        df = pd.read_sql_table(table_name="test_" + table, con=db.engine)
                        print(df)
                        print()

                    _test_db_consistency(self, db)

                    # Check that we have the correct table names.
                    with db.engine.begin() as conn:
                        df = pd.read_sql("SELECT * FROM test_alembic_version", conn)
                        print(df)


                    # self.assertTrue(db.engine.url.database.startswith("test_"))


    def test_copy(self):
        pass

    def test_migrate_prefix(self):
        pass

class TestDbV2Migration(TestCase):
    """Migrations from legacy sqlite db to sqlalchemy-managed databases of
    various kinds.
    """

    def test_db_migration_sqlite_file(self):
        """Test migration from legacy sqlite db to sqlite db."""
        with clean_db("sqlite_file") as db:
            _test_db_migration(db)

    def test_db_migration_postgres(self):
        """Test migration from legacy sqlite db to postgres db."""
        with clean_db("postgres") as db:
            _test_db_migration(db)

    def test_db_migration_mysql(self):
        """Test migration from legacy sqlite db to mysql db."""
        with clean_db("mysql") as db:
            _test_db_migration(db)

    def test_db_consistency_sqlite_file(self):
        """Test database consistency after migration to sqlite."""
        with clean_db("sqlite_file") as db:
            _test_db_consistency(self, db)

    def test_db_consistency_postgres(self):
        """Test database consistency after migration to postgres."""
        with clean_db("postgres") as db:
            _test_db_consistency(self, db)

    def test_db_consistency_mysql(self):
        """Test database consistency after migration to mysql."""
        with clean_db("mysql") as db:
            _test_db_consistency(self, db)

    def test_future_db(self):
        """Check handling of database that is newer than the current
        trulens_eval's db version. 
        
        We expect a warning and exception."""

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
        """Migration from non-latest lagecy db files all the way to v2 database.

        This involves migrating the legacy dbs to the latest legacy first.
        """

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
    """Provider for testing purposes."""

    def length(self, text: str) -> float:  # noqa
        """Test feedback that does nothing except return length of input"""

        return float(len(text))


@contextmanager
def clean_db(
    alias: str,
    **kwargs: Dict[str, Any]
) -> Iterator[SQLAlchemyDB]:
    """Yields a clean database instance for the given database type.
    
    Args:
        alias: Database type to use from the following: `sqlite_file`,
            `sqlite_memory`, `postgres`, `mysql`.

        kwargs: Additional keyword arguments to pass to the database
            constructor.
    """

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # NOTE: The parameters below come from the docker definition in the
        # `trulens_eval/docker/test-database.yaml` file.
        url = {
            "sqlite_memory":
                "sqlite:///:memory:",
                # TODO: Test this one more.
                # NOTE: Sqlalchemy docs say this should be written
                # "sqlite://:memory:" but that gives an error on mac at least.
            "sqlite_file":
                f"sqlite:///{Path(tmp) / 'test.sqlite'}",
            "postgres":
                "postgresql+psycopg2://pg-test-user:pg-test-pswd@localhost/pg-test-db",
            "mysql":
                "mysql+pymysql://mysql-test-user:mysql-test-pswd@localhost/mysql-test-db",
        }[alias]

        db = SQLAlchemyDB.from_db_url(url, **kwargs)

        # NOTE(piotrm): I couldn't figure out why these things were done here.
        #downgrade_db(
        #    db.engine, revision="base"
        #)  # drops all tables except `db.version_table`
        #with db.engine.connect() as conn:
        #    conn.execute(text(f"DROP TABLE {db.table_prefix}version_table"))

        yield db


def assert_revision(
    engine: Engine, expected: Union[None, str], status: Literal["in_sync",
                                                                "behind"]
):
    """Asserts that the version of the database `engine` is `expected` and
    has the `status` flag set."""

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


def _test_db_consistency(test: TestCase, db: SQLAlchemyDB):
    db.migrate_database()  # ensure latest revision

    _populate_data(db)

    with db.session.begin() as session:
        # delete the only app
        session.delete(session.query(db.orm.AppDefinition).one())

        # records are deleted in cascade
        test.assertEqual(
            session.query(db.orm.Record).all(),
            [],
            "Expected no records."
        )

        # feedbacks results are deleted in cascade
        test.assertEqual(
            session.query(db.orm.FeedbackResult).all(),
            [],
            "Expected no feedback results."
        )

        # feedback defs are preserved
        test.assertEqual(
            len(session.query(db.orm.FeedbackDefinition).all()), 1,
            "Expected exactly one feedback to be in the db."
        )

    _populate_data(db)

    with db.session.begin() as session:
        test.assertEqual(
            len(session.query(db.orm.Record).all()), 1,
            "Expected exactly one record."
        )

        test.assertEqual(
            len(session.query(db.orm.FeedbackResult).all()), 1,
            "Expected exactly one feedback result."
        )

        ress = session.query(db.orm.FeedbackResult).all()
        for res in ress:
            print("result record=", res.record)
            print("result def=", res.feedback_definition)

        # delete the only record
        session.delete(session.query(db.orm.Record).one())

        ress = session.query(db.orm.FeedbackResult).all()
        for res in ress:
            print("result record=", res.record)
            print("result def=", res.feedback_definition)

        # feedbacks results are deleted in cascade
        test.assertEqual(session.query(db.orm.FeedbackResult).all(), [], "Expected no feedback results.")

        # apps are preserved
        test.assertTrue(session.query(db.orm.AppDefinition).one(), "Expected an app.")

        # feedback defs are preserved
        test.assertTrue(session.query(db.orm.FeedbackDefinition).one(), "Expected a feedback definition.")


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
