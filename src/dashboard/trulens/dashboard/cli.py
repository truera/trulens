"""Command-line interface for launching the TruLens dashboard.

The entry point is installed as `trulens-dashboard`. With no arguments it looks
for a TruLens database in the current directory and serves it:

```bash
trulens-dashboard
```
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

import sqlalchemy as sa
from trulens.core.database import base as core_db

logger = logging.getLogger(__name__)

_SQLITE_SUFFIXES: Tuple[str, ...] = (".sqlite", ".sqlite3", ".db")
"""File suffixes considered when scanning a directory for a local database."""

_VERSION_TABLE_SUFFIX = "alembic_version"

_ENV_DATABASE_URL = "TRULENS_DATABASE_URL"


class DatabaseNotFoundError(Exception):
    """No TruLens database could be located."""


def _url_for_file(path: Path) -> str:
    """Build a SQLAlchemy SQLite url for a local file."""

    return f"sqlite:///{path}"


def _table_names(url: str) -> List[str]:
    """List table names for a database url, or an empty list if unreadable."""

    try:
        engine = sa.create_engine(url)
    except Exception as exc:
        logger.debug("Could not create engine for %s: %s", url, exc)
        return []

    try:
        return list(sa.inspect(engine).get_table_names())
    except Exception as exc:
        logger.debug("Could not inspect %s: %s", url, exc)
        return []
    finally:
        engine.dispose()


def detect_database_prefix(url: str) -> Optional[str]:
    """Infer the table prefix used by a TruLens database.

    Args:
        url: SQLAlchemy url of the database to inspect.

    Returns:
        The detected prefix, or `None` if the database does not look like a
            TruLens database.
    """

    tables = _table_names(url)
    if not tables:
        return None

    prefixes = [
        table[: -len(_VERSION_TABLE_SUFFIX)]
        for table in tables
        if table.endswith(_VERSION_TABLE_SUFFIX)
    ]

    for prefix in prefixes:
        # An alembic version table alone is ambiguous: other tools use alembic
        # too. Require a TruLens table alongside it.
        if f"{prefix}apps" in tables or f"{prefix}records" in tables:
            return prefix

    if f"{core_db.DEFAULT_DATABASE_PREFIX}records" in tables:
        return core_db.DEFAULT_DATABASE_PREFIX

    return None


def _candidate_files(directory: Path) -> List[Path]:
    """Database files in `directory`, with the default filename ranked first."""

    default_file = directory / core_db.DEFAULT_DATABASE_FILE

    others = sorted(
        path
        for path in directory.glob("*")
        if path.is_file()
        and path.suffix in _SQLITE_SUFFIXES
        and path != default_file
    )

    if default_file.is_file():
        return [default_file, *others]
    return others


def find_database(directory: Path) -> Tuple[str, str]:
    """Find a TruLens database in `directory`.

    The default database filename is preferred; remaining SQLite files are
    checked in alphabetical order. Files that do not contain TruLens tables are
    skipped.

    Args:
        directory: Directory to scan.

    Returns:
        A tuple of the database url and its table prefix.

    Raises:
        DatabaseNotFoundError: No TruLens database was found in `directory`.
    """

    candidates = _candidate_files(directory)

    for path in candidates:
        url = _url_for_file(path)
        prefix = detect_database_prefix(url)
        if prefix is not None:
            return url, prefix

    if candidates:
        listed = ", ".join(path.name for path in candidates)
        raise DatabaseNotFoundError(
            f"No TruLens database found in {directory}. Inspected: {listed}. "
            "None of these contain TruLens tables. Pass --database-url to "
            "point at a database explicitly."
        )

    raise DatabaseNotFoundError(
        f"No TruLens database found in {directory}. Expected "
        f"{core_db.DEFAULT_DATABASE_FILE} or another SQLite file. Run your "
        "instrumented app first, or pass --database-url."
    )


def resolve_database(
    directory: Path,
    database_url: Optional[str] = None,
    database_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the database to serve.

    Precedence is explicit `database_url`, then the `TRULENS_DATABASE_URL`
    environment variable, then discovery within `directory`.

    Args:
        directory: Directory to scan when no url is supplied.

        database_url: Explicit database url.

        database_prefix: Explicit table prefix. Detected when not given.

    Returns:
        A tuple of the database url and its table prefix.
    """

    url = database_url or os.environ.get(_ENV_DATABASE_URL) or None

    if url is None:
        found_url, found_prefix = find_database(directory)
        return found_url, database_prefix or found_prefix

    if database_prefix is not None:
        return url, database_prefix

    return url, detect_database_prefix(url) or core_db.DEFAULT_DATABASE_PREFIX


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trulens-dashboard",
        description=(
            "Launch the TruLens dashboard against a local database. With no "
            "arguments, looks for a TruLens database in the current directory."
        ),
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        default=".",
        type=Path,
        help="Directory to search for a database. Defaults to the current directory.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "Database url to serve, e.g. sqlite:///default.sqlite. Skips "
            f"discovery. Defaults to ${_ENV_DATABASE_URL} when set."
        ),
    )
    parser.add_argument(
        "--database-prefix",
        default=None,
        help="Table prefix. Detected from the database when omitted.",
    )
    parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="Port to serve on. An unused port is chosen when omitted.",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Address to bind to.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Stop any dashboard already running before starting.",
    )
    parser.add_argument(
        "--find",
        action="store_true",
        help="Print the database that would be served and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the `trulens-dashboard` command."""

    args = _parser().parse_args(argv)

    directory = args.directory.expanduser().resolve()

    try:
        url, prefix = resolve_database(
            directory,
            database_url=args.database_url,
            database_prefix=args.database_prefix,
        )
    except DatabaseNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    if args.find:
        sys.stdout.write(f"database url: {url}\ntable prefix: {prefix}\n")
        return 0

    # Imported here so that `--find` and `--help` stay fast and do not require
    # a working session or streamlit install.
    from trulens.core import session as core_session
    from trulens.dashboard import run as dashboard_run

    try:
        session = core_session.TruSession(
            database_url=url, database_prefix=prefix
        )
    except Exception as exc:
        sys.stderr.write(f"Could not open {url}: {exc}\n")
        return 1

    try:
        proc = dashboard_run.run_dashboard(
            session=session,
            port=args.port,
            address=args.address,
            force=args.force,
        )
    except Exception as exc:
        sys.stderr.write(f"Could not start dashboard: {exc}\n")
        return 1

    try:
        return proc.wait()
    except KeyboardInterrupt:
        dashboard_run.stop_dashboard(session=session)
        return 0


if __name__ == "__main__":
    sys.exit(main())
