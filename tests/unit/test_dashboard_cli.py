"""Unit tests for the `trulens-dashboard` CLI database discovery."""

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from trulens.core.database import base as core_db
from trulens.dashboard import cli as dashboard_cli

from tests.test import TruTestCase

_PREFIX = core_db.DEFAULT_DATABASE_PREFIX


def _make_trulens_db(path: Path, prefix: str = _PREFIX) -> None:
    """Create a SQLite file with the tables that mark a TruLens database."""

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"create table {prefix}alembic_version (version_num text)"
        )
        connection.execute(f"create table {prefix}apps (app_id text)")
        connection.execute(f"create table {prefix}records (record_id text)")
        connection.commit()
    finally:
        connection.close()


def _make_other_db(path: Path) -> None:
    """Create a SQLite file that is not a TruLens database."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("create table alembic_version (version_num text)")
        connection.execute("create table unrelated (a int)")
        connection.commit()
    finally:
        connection.close()


class TestDashboardCliDiscovery(TruTestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self._tempdir.name)
        # Discovery consults this variable; keep tests isolated from the shell.
        os.environ.pop("TRULENS_DATABASE_URL", None)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_finds_default_database_file(self):
        """The default filename is discovered without any arguments."""

        db_path = self.directory / core_db.DEFAULT_DATABASE_FILE
        _make_trulens_db(db_path)

        url, prefix = dashboard_cli.find_database(self.directory)

        self.assertEqual(url, f"sqlite:///{db_path}")
        self.assertEqual(prefix, _PREFIX)

    def test_finds_non_default_filename(self):
        """A TruLens database under another name is still discovered."""

        db_path = self.directory / "my_evals.sqlite"
        _make_trulens_db(db_path)

        url, prefix = dashboard_cli.find_database(self.directory)

        self.assertEqual(url, f"sqlite:///{db_path}")
        self.assertEqual(prefix, _PREFIX)

    def test_default_filename_preferred(self):
        """The default filename wins over other candidates."""

        default_path = self.directory / core_db.DEFAULT_DATABASE_FILE
        _make_trulens_db(default_path)
        # Sorts before the default name alphabetically.
        _make_trulens_db(self.directory / "aaa.sqlite")

        url, _ = dashboard_cli.find_database(self.directory)

        self.assertEqual(url, f"sqlite:///{default_path}")

    def test_detects_custom_prefix(self):
        """A non-default table prefix is detected from the schema."""

        db_path = self.directory / core_db.DEFAULT_DATABASE_FILE
        _make_trulens_db(db_path, prefix="custom_")

        _, prefix = dashboard_cli.find_database(self.directory)

        self.assertEqual(prefix, "custom_")

    def test_skips_non_trulens_database(self):
        """A SQLite file without TruLens tables is not selected."""

        _make_other_db(self.directory / "other.sqlite")
        db_path = self.directory / "real.sqlite"
        _make_trulens_db(db_path)

        url, _ = dashboard_cli.find_database(self.directory)

        self.assertEqual(url, f"sqlite:///{db_path}")

    def test_raises_when_directory_empty(self):
        with self.assertRaises(dashboard_cli.DatabaseNotFoundError):
            dashboard_cli.find_database(self.directory)

    def test_raises_when_no_trulens_database(self):
        _make_other_db(self.directory / "other.sqlite")

        with self.assertRaises(dashboard_cli.DatabaseNotFoundError) as context:
            dashboard_cli.find_database(self.directory)

        # The message should name what was inspected so the user can tell why.
        self.assertIn("other.sqlite", str(context.exception))

    def test_explicit_url_skips_discovery(self):
        """An explicit url is used even when the directory has a database."""

        _make_trulens_db(self.directory / core_db.DEFAULT_DATABASE_FILE)
        other = self.directory / "explicit.sqlite"
        _make_trulens_db(other, prefix="custom_")

        url, prefix = dashboard_cli.resolve_database(
            self.directory, database_url=f"sqlite:///{other}"
        )

        self.assertEqual(url, f"sqlite:///{other}")
        self.assertEqual(prefix, "custom_")

    def test_explicit_prefix_overrides_detection(self):
        db_path = self.directory / core_db.DEFAULT_DATABASE_FILE
        _make_trulens_db(db_path)

        _, prefix = dashboard_cli.resolve_database(
            self.directory, database_prefix="override_"
        )

        self.assertEqual(prefix, "override_")

    def test_env_var_used_when_no_argument(self):
        db_path = self.directory / "from_env.sqlite"
        _make_trulens_db(db_path)
        empty = self.directory / "empty"
        empty.mkdir()

        with mock.patch.dict(
            os.environ, {"TRULENS_DATABASE_URL": f"sqlite:///{db_path}"}
        ):
            url, prefix = dashboard_cli.resolve_database(empty)

        self.assertEqual(url, f"sqlite:///{db_path}")
        self.assertEqual(prefix, _PREFIX)

    def test_explicit_url_beats_env_var(self):
        explicit = self.directory / "explicit.sqlite"
        _make_trulens_db(explicit)

        with mock.patch.dict(
            os.environ, {"TRULENS_DATABASE_URL": "sqlite:///ignored.sqlite"}
        ):
            url, _ = dashboard_cli.resolve_database(
                self.directory, database_url=f"sqlite:///{explicit}"
            )

        self.assertEqual(url, f"sqlite:///{explicit}")

    def test_unreadable_url_falls_back_to_default_prefix(self):
        """An unreachable database does not crash prefix detection."""

        prefix = dashboard_cli.detect_database_prefix(
            f"sqlite:///{self.directory / 'missing.sqlite'}"
        )

        self.assertIsNone(prefix)


class TestDashboardCliMain(TruTestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self._tempdir.name)
        os.environ.pop("TRULENS_DATABASE_URL", None)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_find_flag_reports_database(self):
        db_path = self.directory / core_db.DEFAULT_DATABASE_FILE
        _make_trulens_db(db_path)

        code = dashboard_cli.main(["--find", "--dir", str(self.directory)])

        self.assertEqual(code, 0)

    def test_find_flag_returns_error_when_missing(self):
        code = dashboard_cli.main(["--find", "--dir", str(self.directory)])

        self.assertEqual(code, 1)

    def test_launches_dashboard_with_discovered_database(self):
        """`main` wires the discovered database into `run_dashboard`."""

        _make_trulens_db(self.directory / core_db.DEFAULT_DATABASE_FILE)

        fake_session = mock.Mock(name="TruSession")
        fake_proc = mock.Mock(name="proc")
        fake_proc.wait.return_value = 0

        with mock.patch(
            "trulens.core.session.TruSession", return_value=fake_session
        ) as session_ctor:
            with mock.patch(
                "trulens.dashboard.run.run_dashboard", return_value=fake_proc
            ) as run_dashboard:
                code = dashboard_cli.main([
                    "--dir",
                    str(self.directory),
                    "--port",
                    "1234",
                ])

        self.assertEqual(code, 0)
        # `main` resolves the directory, so compare against the resolved path
        # (on macOS /var is a symlink to /private/var).
        resolved = self.directory.resolve() / core_db.DEFAULT_DATABASE_FILE
        session_ctor.assert_called_once_with(
            database_url=f"sqlite:///{resolved}", database_prefix=_PREFIX
        )
        _, kwargs = run_dashboard.call_args
        self.assertIs(kwargs["session"], fake_session)
        self.assertEqual(kwargs["port"], 1234)

    def test_returns_error_when_dashboard_fails(self):
        _make_trulens_db(self.directory / core_db.DEFAULT_DATABASE_FILE)

        with mock.patch("trulens.core.session.TruSession"):
            with mock.patch(
                "trulens.dashboard.run.run_dashboard",
                side_effect=RuntimeError("boom"),
            ):
                code = dashboard_cli.main(["--dir", str(self.directory)])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
