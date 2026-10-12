"""Tests for repeated `TruSession(...)` calls on the existing singleton."""

import tempfile

from trulens.core import session as core_session
from trulens.core.database import connector as core_connector

from tests import test as mod_test

_LOGGER = "trulens.core.session"


class TestSessionReinit(mod_test.TruTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.url_a = f"sqlite:///{tmp.name}/a.sqlite"
        self.url_b = f"sqlite:///{tmp.name}/b.sqlite"
        self.session = core_session.TruSession(database_url=self.url_a)

    def _url_in_use(self) -> str:
        return str(self.session.connector.db.engine.url)

    def test_different_database_url_warns_and_keeps_session(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            again = core_session.TruSession(database_url=self.url_b)

        self.assertIs(again, self.session)
        self.assertEqual(self._url_in_use(), self.url_a)
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn(self.url_b, message)
        self.assertIn("TruSession.delete_singleton(TruSession)", message)

    def test_same_mismatch_warns_once(self):
        with self.assertLogs(_LOGGER, level="WARNING"):
            core_session.TruSession(database_url=self.url_b)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            core_session.TruSession(database_url=self.url_b)

    def test_different_connector_warns(self):
        connector = core_connector.DefaultDBConnector(database_url=self.url_b)
        with self.assertLogs(_LOGGER, level="WARNING"):
            again = core_session.TruSession(connector=connector)

        self.assertIs(again, self.session)
        self.assertIsNot(self.session.connector, connector)

    def test_no_arguments_does_not_warn(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            again = core_session.TruSession()

        self.assertIs(again, self.session)

    def test_same_settings_do_not_warn(self):
        equal_connector = core_connector.DefaultDBConnector(
            database_url=self.url_a
        )
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            core_session.TruSession(database_url=self.url_a)
            core_session.TruSession(connector=self.session.connector)
            core_session.TruSession(connector=equal_connector)
            core_session.TruSession(
                database_url=self.url_a,
                database_prefix=self.session.connector.db.table_prefix,
            )

        self.assertEqual(self._url_in_use(), self.url_a)

    def test_warning_hides_database_password(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            core_session.TruSession(
                database_url="postgresql://user:hunter2@localhost/evals"
            )

        message = logs.records[0].getMessage()
        self.assertIn("localhost/evals", message)
        self.assertNotIn("hunter2", message)
