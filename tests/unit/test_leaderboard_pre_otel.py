"""Regression test: get_leaderboard_aggregates must work with OTel tracing
disabled.

`cost_json` is a Text column (see `TYPE_JSON = Text` in
`trulens.core.database.orm`), so the pre-OTel aggregation path cannot use
JSON-style `[]` subscript indexing on it - SQLAlchemy raises
`NotImplementedError: Operator 'getitem' is not supported on this
expression` while building the statement, before any database call is made.
This affected every backend unconditionally.
"""

import os

# CRITICAL: Set OTEL environment variable BEFORE any TruLens imports
# This must be done before any TruSession is created to avoid freezing the experimental flag
os.environ["TRULENS_OTEL_TRACING"] = "0"

from trulens.core import session as core_session
from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import record as record_schema

from tests.test import TruTestCase


class TestLeaderboardAggregatesPreOtel(TruTestCase):
    def setUp(self):
        self.session = core_session.TruSession()
        self.session.reset_database()

    def _add_record(self, app_id, n_tokens, cost):
        self.session.add_record(
            record_schema.Record(
                app_id=app_id,
                calls=[],
                cost=base_schema.Cost(
                    n_tokens=n_tokens, cost=cost, cost_currency="USD"
                ),
            )
        )

    def test_get_leaderboard_aggregates_does_not_raise(self):
        app = app_schema.AppDefinition(
            app_name="pre_otel_leaderboard_app",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )
        app_id = self.session.add_app(app)

        self._add_record(app_id, n_tokens=10, cost=0.01)
        self._add_record(app_id, n_tokens=20, cost=0.03)

        # Prior to the fix this raised NotImplementedError while building
        # the SQL statement, before touching the database.
        leaderboard, _ = self.session.connector.db.get_leaderboard_aggregates()

        row = leaderboard[leaderboard["app_id"] == app_id].iloc[0]
        self.assertEqual(row["Records"], 2)
        self.assertAlmostEqual(row["Total Tokens"], 15.0)  # avg(10, 20)
        self.assertAlmostEqual(row["Total Cost (USD)"], 0.04)  # sum(0.01, 0.03)
        # `Record.latency` doesn't exist as a column at all (only
        # `perf_json` does) - this was a second, unreachable bug hiding
        # behind the reported one, only surfaced by actually running the
        # fixed query rather than trusting the original repro's stack trace.
        self.assertIn("Average Latency (s)", row.index)

    def test_get_leaderboard_aggregates_empty_database(self):
        leaderboard, feedback_cols = (
            self.session.connector.db.get_leaderboard_aggregates()
        )
        self.assertTrue(leaderboard.empty)
        self.assertEqual(feedback_cols, [])


if __name__ == "__main__":
    import unittest

    unittest.main()
