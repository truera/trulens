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
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, Literal, Union
from unittest import main
from unittest import TestCase

import pandas as pd
from sqlalchemy import Engine
from trulens.core import Tru
from trulens.core import TruBasicApp
from trulens.core.database.base import DB
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.migrations import DbRevisions
from trulens.core.database.migrations import downgrade_db
from trulens.core.database.migrations import get_revision_history
from trulens.core.database.migrations import upgrade_db
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.database.utils import copy_database
from trulens.core.database.utils import is_legacy_sqlite
from trulens.core.feedback import Feedback
from trulens.core.feedback import Provider
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.feedback import Select


class TestDBSpecifications(TestCase):
    """Tests for database options."""

    def test_prefix(self):
        """Test that the table prefix is correctly used to name tables in the database."""

        db_types = ["sqlite_file"]  # , "postgres", "mysql", "sqlite_memory"
        # sqlite_memory might have problems with multithreading of tests

        for db_type in db_types:
            with self.subTest(msg=f"prefix for {db_type}"):
                with clean_db(db_type, table_prefix="test_") as db:
                    _test_db_consistency(self, db)

                    # Check that we have the correct table names.
                    with db.engine.begin() as conn:
                        df = pd.read_sql(
                            "SELECT * FROM test_alembic_version", conn
                        )
                        print(df)

    def test_copy(self):
        """Test copying of databases via [copy_database][trulens_eval.database.utils.copy_database]."""

        db_types = ["sqlite_file"]  # , "postgres", "mysql", "sqlite_memory"
        # sqlite_memory might have problems with multithreading of tests

        for source_db_type in db_types:
            with self.subTest(msg=f"source prefix for {source_db_type}"):
                with clean_db(
                    source_db_type, table_prefix="test_prior_"
                ) as db_prior:
                    _populate_data(db_prior)

                    for target_db_type in db_types:
                        with self.subTest(
                            msg=f"target prefix for {target_db_type}"
                        ):
                            with clean_db(
                                target_db_type, table_prefix="test_post_"
                            ) as db_post:
                                # This makes the database tables:
                                db_post.migrate_database()

                                # assert database is empty before copying
                                with db_post.session.begin() as session:
                                    for orm_class in [
                                        db_post.orm.AppDefinition,
                                        db_post.orm.FeedbackDefinition,
                                        db_post.orm.Record,
                                        db_post.orm.FeedbackResult,
                                    ]:
                                        self.assertEqual(
                                            session.query(orm_class).all(),
                                            [],
                                            f"Expected no {orm_class}.",
                                        )

                                copy_database(
                                    src_url=db_prior.engine.url,
                                    tgt_url=db_post.engine.url,
                                    src_prefix="test_prior_",
                                    tgt_prefix="test_post_",
                                )

                                # assert database contains exactly one of each row
                                with db_post.session.begin() as session:
                                    for orm_class in [
                                        db_post.orm.AppDefinition,
                                        db_post.orm.FeedbackDefinition,
                                        db_post.orm.Record,
                                        db_post.orm.FeedbackResult,
                                    ]:
                                        self.assertEqual(
                                            len(session.query(orm_class).all()),
                                            1,
                                            f"Expected exactly one {orm_class}.",
                                        )

    def test_migrate_prefix(self):
        """Test that database migration works across different prefixes."""

        db_types = ["sqlite_file"]  # , "postgres", "mysql", "sqlite_memory"
        # sqlite_memory might have problems with multithreading of tests

        for db_type in db_types:
            with self.subTest(msg=f"prefix for {db_type}"):
                with clean_db(db_type, table_prefix="test_prior_") as db_prior:
                    _test_db_consistency(self, db_prior)

                    # Migrate the database.
                    with clean_db(
                        db_type, table_prefix="test_post_"
                    ) as db_post:
                        db_post.migrate_database(prior_prefix="test_prior_")

                        # Check that we have the correct table names.
                        with db_post.engine.begin() as conn:
                            df = pd.read_sql(
                                "SELECT * FROM test_post_alembic_version", conn
                            )
                            print(df)

                        _test_db_consistency(self, db_post)


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

        for folder in (
            Path(__file__).parent.parent.parent / "release_dbs"
        ).iterdir():
            _dbfile = folder / "default.sqlite"

            if "infty" not in str(folder):
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

        for folder in (
            Path(__file__).parent.parent.parent / "release_dbs"
        ).iterdir():
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


class MockFeedback(Provider):
    """Provider for testing purposes."""

    def length(self, text: str) -> float:  # noqa
        """Test feedback that does nothing except return length of input"""

        return float(len(text))


@contextmanager
def clean_db(alias: str, **kwargs: Dict[str, Any]) -> Iterator[SQLAlchemyDB]:
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
            "sqlite_memory": "sqlite:///:memory:",
            # TODO: Test this one more.
            # NOTE: Sqlalchemy docs say this should be written
            # "sqlite://:memory:" but that gives an error on mac at least.
            "sqlite_file": f"sqlite:///{Path(tmp) / 'test.sqlite'}",
            "postgres": "postgresql+psycopg2://pg-test-user:pg-test-pswd@localhost/pg-test-db",
            "mysql": "mysql+pymysql://mysql-test-user:mysql-test-pswd@localhost/mysql-test-db",
        }[alias]

        db = SQLAlchemyDB.from_db_url(url, **kwargs)

        # NOTE(piotrm): I couldn't figure out why these things were done here.
        # downgrade_db(
        #    db.engine, revision="base"
        # )  # drops all tables except `db.version_table`
        # with db.engine.connect() as conn:
        #    conn.execute(text(f"DROP TABLE {db.table_prefix}version_table"))

        yield db


def assert_revision(
    engine: Engine,
    expected: Union[None, str],
    status: Literal["in_sync", "behind"],
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
        assert (
            int(next_rev) == i + 1
        ), f"Versions must be monotonically increasing from 1: {history}"
        assert_revision(engine, curr_rev, "behind")
        upgrade_db(engine, revision=next_rev)
        curr_rev = next_rev

    # validate final state
    assert_revision(engine, history[-1], "in_sync")

    # apply all downgrades
    downgrade_db(engine, revision="base")
    assert_revision(engine, None, "behind")


def debug_dump(db: SQLAlchemyDB):
    """Debug function to dump all tables in the database."""

    print("  # registry")
    for n, t in db.orm.registry.items():
        print("   ", n, t)

    with db.session.begin() as session:
        print("  # feedback_def")
        ress = session.query(db.orm.FeedbackDefinition).all()
        for res in ress:
            print("    feedback_def", res.feedback_definition_id)

        print("  # app")
        ress = session.query(db.orm.AppDefinition).all()
        for res in ress:
            print("    app", res.app_id)  # no feedback results
            for subres in res.records:
                print("      record", subres.record_id)

        print("  # record")
        ress = session.query(db.orm.Record).all()
        for res in ress:
            print("    record", res.record_id)
            for subres in res.feedback_results:
                print("      feedback_result", subres.feedback_result_id)

        print("  # feedback")
        ress = session.query(db.orm.FeedbackResult).all()
        for res in ress:
            print(
                "    feedback_result",
                res.feedback_result_id,
                res.feedback_definition,
            )


def _test_db_consistency(test: TestCase, db: SQLAlchemyDB):
    db.migrate_database()  # ensure latest revision

    _populate_data(db)

    print("### before delete app:")
    debug_dump(db)

    with db.session.begin() as session:
        # delete the only app
        session.delete(session.query(db.orm.AppDefinition).one())

        # records are deleted in cascade
        test.assertEqual(
            session.query(db.orm.Record).all(), [], "Expected no records."
        )

        # feedbacks results are deleted in cascade
        test.assertEqual(
            session.query(db.orm.FeedbackResult).all(),
            [],
            "Expected no feedback results.",
        )

        # feedback defs are preserved
        test.assertEqual(
            len(session.query(db.orm.FeedbackDefinition).all()),
            1,
            "Expected exactly one feedback to be in the db.",
        )

    _populate_data(db)

    print("### before delete record:")
    debug_dump(db)

    with db.session.begin() as session:
        test.assertEqual(
            len(session.query(db.orm.Record).all()),
            1,
            "Expected exactly one record.",
        )

        test.assertEqual(
            len(session.query(db.orm.FeedbackResult).all()),
            1,
            "Expected exactly one feedback result.",
        )

        # delete the only record
        session.delete(session.query(db.orm.Record).one())

        # feedbacks results are deleted in cascade
        test.assertEqual(
            session.query(db.orm.FeedbackResult).all(),
            [],
            "Expected no feedback results.",
        )

        # apps are preserved
        test.assertEqual(
            len(session.query(db.orm.AppDefinition).all()),
            1,
            "Expected an app.",
        )

        # feedback defs are preserved. Note that this requires us to use the
        # same feedback_definition_id in _populate_data.
        test.assertEqual(
            len(session.query(db.orm.FeedbackDefinition).all()),
            1,
            "Expected a feedback definition.",
        )


def _populate_data(db: DB):
    tru = Tru()
    tru.db = (
        db  # because of the singleton behavior, db must be changed manually
    )

    fb = Feedback(
        imp=MockFeedback().length,
        feedback_definition_id="mock",
        selectors={"text": Select.RecordOutput},
    )
    app = TruBasicApp(
        text_to_text=lambda x: x,
        # app_id="test",
        db=db,
        feedbacks=[fb],
        feedback_mode=FeedbackMode.WITH_APP_THREAD,
    )
    _, rec = app.with_record(app.app.__call__, "boo")

    print("waiting for feedback results")
    for res in rec.wait_for_feedback_results():
        print("  ", res)

    return fb, app, rec


if __name__ == "__main__":
    main()
