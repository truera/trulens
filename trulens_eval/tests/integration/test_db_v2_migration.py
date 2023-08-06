from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from trulens_eval import Tru
from trulens_eval.db import LocalSQLite
from trulens_eval.db_v2.db import SqlAlchemyDB, is_legacy_sqlite
from trulens_eval.db_v2.migrations import migrate_db, DbRevisions, get_revision_history


class TestDbV2Migration(TestCase):

    def test_db_v2_migrations(self):
        for url in [
            "sqlite:///:memory:"  # warn: built-in sqlite is not threadsafe when used in-memory
        ]:
            db = SqlAlchemyDB.from_db_url(url)
            engine = db.engine
            history = get_revision_history(engine)
            curr_rev = None

            # apply each upgrade at a time
            for i, next_rev in enumerate(history):
                assert int(next_rev) == i + 1, \
                    f"Versions must be monotonically increasing: {history}"
                revisions = DbRevisions.load(engine)
                assert revisions.current == curr_rev
                assert revisions.behind
                migrate_db(engine, revision=next_rev)
                curr_rev = next_rev

            # validate final state
            revisions = DbRevisions.load(engine)
            assert revisions.current == history[-1]
            assert revisions.in_sync

            # apply all downgrades
            migrate_db(engine, revision="base")
            revisions = DbRevisions.load(engine)
            assert revisions.current == history[0]

    def test_migrate_legacy_sqlite_file(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp).joinpath("legacy.sqlite")

            # trigger database creation
            LocalSQLite(filename=file)
            assert file.exists() and file.is_file()

            # run migration
            tru = Tru(database_file=str(file))
            assert is_legacy_sqlite(tru.db.engine)
            tru.migrate_database()

            # validate final state
            assert not is_legacy_sqlite(tru.db.engine)
            assert DbRevisions.load(tru.db.engine).in_sync


if __name__ == '__main__':
    main()
