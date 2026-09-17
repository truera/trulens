"""Regression tests for _get_leaderboard_aggregates_pre_otel.

Verifies that the leaderboard aggregation path used when OTEL tracing is
disabled does not raise NotImplementedError due to SQLAlchemy [] subscript
being unsupported on TEXT columns (cost_json, perf_json are TYPE_JSON = Text).
"""

import datetime
import json
import unittest

from tests.test import TruTestCase


class TestLeaderboardPreOtel(TruTestCase):
    """Regression test: cost_json/perf_json are TEXT, not JSON columns."""

    def _make_session(self):
        from trulens.core.session import TruSession

        return TruSession(database_url="sqlite:///:memory:")

    def test_leaderboard_empty_db_does_not_raise(self):
        """An empty database returns two empty structures without crashing."""
        session = self._make_session()
        db = session.connector.db

        df, feedback_cols = db._get_leaderboard_aggregates_pre_otel()

        self.assertIsNotNone(df)
        self.assertListEqual(feedback_cols, [])
        self.assertTrue(df.empty)

    def test_leaderboard_with_records_aggregates_correctly(self):
        """Leaderboard aggregation works end-to-end with real SQLite records."""
        tru_session = self._make_session()
        db = tru_session.connector.db

        app_id = "test_app_v1"
        app_json = json.dumps({"app_name": "test_app", "app_version": "v1"})
        with db.session.begin() as s:
            app_row = db.orm.AppDefinition(
                app_id=app_id,
                app_name="test_app",
                app_version="v1",
                app_json=app_json,
            )
            s.add(app_row)

        cost1 = json.dumps({
            "n_tokens": 100,
            "n_prompt_tokens": 60,
            "n_completion_tokens": 40,
            "cost": 0.01,
            "cost_currency": "USD",
        })
        cost2 = json.dumps({
            "n_tokens": 200,
            "n_prompt_tokens": 120,
            "n_completion_tokens": 80,
            "cost": 0.02,
            "cost_currency": "USD",
        })
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        perf = json.dumps({
            "start_time": now.isoformat(),
            "end_time": (now + datetime.timedelta(seconds=2)).isoformat(),
        })

        with db.session.begin() as s:
            for rid, cost in [("rec1", cost1), ("rec2", cost2)]:
                s.add(
                    db.orm.Record(
                        record_id=rid,
                        app_id=app_id,
                        input="q",
                        output="a",
                        record_json="{}",
                        cost_json=cost,
                        perf_json=perf,
                        ts=now.timestamp(),
                        tags="",
                    )
                )

        df, feedback_cols = db._get_leaderboard_aggregates_pre_otel()

        self.assertFalse(df.empty, "Expected non-empty DataFrame")
        self.assertIn("Records", df.columns)
        self.assertIn("Total Tokens", df.columns)
        self.assertIn("Average Latency (s)", df.columns)
        self.assertIn("Total Cost (USD)", df.columns)

        row = df.iloc[0]
        self.assertEqual(int(row["Records"]), 2)
        self.assertAlmostEqual(float(row["Total Tokens"]), 300, places=1)
        self.assertAlmostEqual(float(row["Total Cost (USD)"]), 0.03, places=4)
        self.assertGreater(float(row["Average Latency (s)"]), 0)
        self.assertListEqual(feedback_cols, [])

    def test_leaderboard_filter_by_app_name(self):
        """app_name filter returns only matching apps."""
        tru_session = self._make_session()
        db = tru_session.connector.db

        cost = json.dumps({
            "n_tokens": 50,
            "cost": 0.005,
            "cost_currency": "USD",
        })
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        perf = json.dumps({
            "start_time": now.isoformat(),
            "end_time": (now + datetime.timedelta(seconds=1)).isoformat(),
        })

        for app_name, app_id in [("appA", "appA_v1"), ("appB", "appB_v1")]:
            with db.session.begin() as s:
                s.add(
                    db.orm.AppDefinition(
                        app_id=app_id,
                        app_name=app_name,
                        app_version="v1",
                        app_json=json.dumps({
                            "app_name": app_name,
                            "app_version": "v1",
                        }),
                    )
                )
                s.add(
                    db.orm.Record(
                        record_id=f"rec_{app_id}",
                        app_id=app_id,
                        input="q",
                        output="a",
                        record_json="{}",
                        cost_json=cost,
                        perf_json=perf,
                        ts=now.timestamp(),
                        tags="",
                    )
                )

        df, _ = db._get_leaderboard_aggregates_pre_otel(app_name="appA")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["app_name"], "appA")

    def test_leaderboard_mixed_currency_split(self):
        """USD and Snowflake Credits costs are reported in separate columns."""
        tru_session = self._make_session()
        db = tru_session.connector.db

        app_id = "mixed_app_v1"
        with db.session.begin() as s:
            s.add(
                db.orm.AppDefinition(
                    app_id=app_id,
                    app_name="mixed_app",
                    app_version="v1",
                    app_json='{"app_name": "mixed_app", "app_version": "v1"}',
                )
            )

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        perf = json.dumps({
            "start_time": now.isoformat(),
            "end_time": (now + datetime.timedelta(seconds=1)).isoformat(),
        })

        records = [
            (
                "rec_usd_1",
                json.dumps({
                    "n_tokens": 100,
                    "cost": 5.0,
                    "cost_currency": "USD",
                }),
            ),
            (
                "rec_usd_2",
                json.dumps({
                    "n_tokens": 200,
                    "cost": 5.0,
                    "cost_currency": "USD",
                }),
            ),
            (
                "rec_sf_1",
                json.dumps({
                    "n_tokens": 0,
                    "cost": 400.0,
                    "cost_currency": "Snowflake credits",
                }),
            ),
            (
                "rec_sf_2",
                json.dumps({
                    "n_tokens": 0,
                    "cost": 600.0,
                    "cost_currency": "Snowflake credits",
                }),
            ),
        ]
        with db.session.begin() as s:
            for rid, cost_json in records:
                s.add(
                    db.orm.Record(
                        record_id=rid,
                        app_id=app_id,
                        input="q",
                        output="a",
                        record_json="{}",
                        cost_json=cost_json,
                        perf_json=perf,
                        ts=now.timestamp(),
                        tags="",
                    )
                )

        df, _ = db._get_leaderboard_aggregates_pre_otel()

        self.assertFalse(df.empty)
        self.assertIn("Total Cost (USD)", df.columns)
        self.assertIn("Total Cost (Snowflake Credits)", df.columns)

        row = df.iloc[0]
        self.assertAlmostEqual(float(row["Total Cost (USD)"]), 10.0, places=4)
        self.assertAlmostEqual(
            float(row["Total Cost (Snowflake Credits)"]), 1000.0, places=4
        )


if __name__ == "__main__":
    unittest.main()
