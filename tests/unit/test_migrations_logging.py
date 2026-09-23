"""Running the database migrations must leave the host application's logging
configuration alone."""

import logging
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import sqlalchemy as sa
from trulens.core.database import migrations as db_migrations


class TestMigrationsLeaveLoggingAlone(TestCase):
    def setUp(self) -> None:
        self.app_logger = logging.getLogger("trulens_tests.host_application")
        root = logging.getLogger()
        saved = (root.level, list(root.handlers))

        def restore() -> None:
            # Undo whatever the migrations did so a failure here does not
            # leak into other tests.
            root.setLevel(saved[0])
            root.handlers[:] = saved[1]
            self.app_logger.disabled = False

        self.addCleanup(restore)

    def test_upgrade_leaves_existing_loggers_and_root_untouched(self) -> None:
        root = logging.getLogger()
        root_level, root_handlers = root.level, list(root.handlers)

        with TemporaryDirectory() as tmp:
            engine = sa.create_engine(
                f"sqlite:///{os.path.join(tmp, 'default.sqlite')}"
            )
            try:
                db_migrations.upgrade_db(engine, revision="head")
            finally:
                engine.dispose()

        self.assertFalse(
            self.app_logger.disabled,
            "a logger created before the migrations must stay enabled",
        )
        self.assertEqual(root.level, root_level)
        self.assertEqual(root.handlers, root_handlers)
