"""
Regression test for https://github.com/truera/trulens/issues/2729.

`get_leaderboard_aggregates` must not crash when OTEL tracing is disabled,
since that is the code path taken by the dashboard leaderboard in the
supported non-OTEL configuration.
"""

import os

# CRITICAL: Set OTEL environment variable BEFORE any TruLens imports
# This must be done before any TruSession is created to avoid freezing the
# experimental flag.
os.environ["TRULENS_OTEL_TRACING"] = "0"

from trulens.apps import custom as custom_app
from trulens.core import session as core_session

from examples.dev.dummy_app.app import DummyApp
from tests.test import TruTestCase


class TestLeaderboardAggregatesPreOtel(TruTestCase):
    @staticmethod
    def setUpClass():
        core_session.TruSession().reset_database()

    def setUp(self):
        self.session = core_session.TruSession()

    def test_get_leaderboard_aggregates_does_not_crash(self):
        app = DummyApp()
        recorder = custom_app.TruCustomApp(
            app, app_name="leaderboard_app", app_version="v1"
        )
        with recorder:
            app.respond_to_query(query="What is the capital of Indonesia?")

        db = self.session.connector.db
        df, feedback_col_names = db.get_leaderboard_aggregates(
            app_name="leaderboard_app"
        )

        self.assertFalse(df.empty)
        self.assertEqual(feedback_col_names, [])
        row = df.iloc[0]
        self.assertEqual(row["Records"], 1)
        self.assertGreater(row["Total Tokens"], 0.0)
        self.assertGreaterEqual(row["Total Cost (USD)"], 0.0)
        self.assertGreaterEqual(row["Average Latency (s)"], 0.0)
