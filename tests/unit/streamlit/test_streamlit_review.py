"""Tests for the dashboard human-review page.

Covers the dashboard flows from truera/trulens#2700: building a queue from the
three presets, previewing before materializing, pulling an item, submitting a
review, and exporting. The page's panels are rendered with `AppTest` against a
real SQLite-backed session, so the flows exercise the same session calls a
reviewer would trigger.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from trulens.core import session as core_session
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.database.connector import default as default_connector
from trulens.core.schema import review as review_schema
from trulens.dashboard.tabs import Review as review_page

RECORDS = pd.DataFrame([
    {
        "record_id": "r1",
        "Groundedness": 0.2,
        "Groundedness direction": True,
        "latency": 2.0,
        "total_cost": 0.01,
        "cost_currency": "USD",
        "app_name": "bot",
        "app_version": "v1",
        "ts": 1.0,
        "input": "q1",
        "output": "a1",
    },
    {
        "record_id": "r2",
        "Groundedness": 0.9,
        "Groundedness direction": True,
        "latency": 20.0,
        "total_cost": 0.20,
        "cost_currency": "USD",
        "app_name": "bot",
        "app_version": "v1",
        "ts": 2.0,
        "input": "q2",
        "output": "a2",
    },
])


def _clear_tru_session_singletons():
    for key in [
        curr
        for curr in core_session.TruSession._singleton_instances
        if curr[0] == "trulens.core.session.TruSession"
    ]:
        del core_session.TruSession._singleton_instances[key]


@pytest.fixture
def session():
    """A TruSession backed by a throwaway SQLite file."""

    with tempfile.TemporaryDirectory() as tempdir:
        db = db_sqlalchemy.SQLAlchemyDB.from_db_url(
            f"sqlite:///{os.path.join(tempdir, 'trulens.sqlite')}"
        )
        db.migrate_database()

        _clear_tru_session_singletons()
        yield core_session.TruSession(
            connector=default_connector.DefaultDBConnector(database=db)
        )
        _clear_tru_session_singletons()


class TestPresetPredicates:
    """The three presets the queue-creation panel offers."""

    def test_feedback_columns_are_those_with_a_direction(self):
        assert review_page._feedback_columns(RECORDS) == ["Groundedness"]

    def test_low_score_preset_threshold(self):
        predicate = review_page._build_predicate(
            RECORDS,
            review_page.PRESET_LOW_SCORE,
            metric="Groundedness",
            threshold=0.5,
            use_top_n=False,
            top_n=10,
            currency=None,
        )
        result = predicate.evaluate(RECORDS)
        assert list(result.mask) == [True, False]

    def test_low_score_preset_top_n(self):
        predicate = review_page._build_predicate(
            RECORDS,
            review_page.PRESET_LOW_SCORE,
            metric="Groundedness",
            threshold=0.5,
            use_top_n=True,
            top_n=1,
            currency=None,
        )
        assert list(predicate.evaluate(RECORDS).mask) == [True, False]

    def test_low_score_preset_needs_a_metric(self):
        assert (
            review_page._build_predicate(
                RECORDS,
                review_page.PRESET_LOW_SCORE,
                metric=None,
                threshold=0.5,
                use_top_n=False,
                top_n=10,
                currency=None,
            )
            is None
        )

    def test_high_latency_preset(self):
        predicate = review_page._build_predicate(
            RECORDS,
            review_page.PRESET_HIGH_LATENCY,
            metric=None,
            threshold=8.0,
            use_top_n=False,
            top_n=10,
            currency=None,
        )
        assert list(predicate.evaluate(RECORDS).mask) == [False, True]

    def test_slowest_preset(self):
        predicate = review_page._build_predicate(
            RECORDS,
            review_page.PRESET_HIGH_LATENCY,
            metric=None,
            threshold=0.0,
            use_top_n=True,
            top_n=1,
            currency=None,
        )
        assert list(predicate.evaluate(RECORDS).mask) == [False, True]

    def test_high_cost_preset(self):
        predicate = review_page._build_predicate(
            RECORDS,
            review_page.PRESET_HIGH_COST,
            metric=None,
            threshold=0.05,
            use_top_n=False,
            top_n=10,
            currency="USD",
        )
        assert list(predicate.evaluate(RECORDS).mask) == [False, True]

    def test_high_cost_preset_needs_a_currency(self):
        assert (
            review_page._build_predicate(
                RECORDS,
                review_page.PRESET_HIGH_COST,
                metric=None,
                threshold=0.05,
                use_top_n=False,
                top_n=10,
                currency=None,
            )
            is None
        )


class TestDashboardFlows:
    """Create, add, pull, review and export, as the page drives them."""

    def test_create_and_pull_and_review_and_export(self, session):
        from trulens.core.review import ReviewTargets

        targets = ReviewTargets.from_records(
            RECORDS,
            where=ReviewTargets.low_score("Groundedness", below=0.5),
            order_by="severity",
            limit=100,
        )
        previewed_ids = [t.target_id for t in targets]

        queue = session.create_review_queue(
            name="low-groundedness",
            instructions="Review these.",
            targets=targets,
        )

        # What the preview showed is what the queue holds.
        items = session.get_review_items(queue.review_queue_id)
        assert [i.target_id for i in items] == previewed_ids

        item = session.claim_next_review_item(queue.review_queue_id)
        assert item.target_id == "r1"
        assert item.selection.selection_reason == "Groundedness < 0.5"

        session.submit_human_review(
            target=item.target,
            verdict="fail",
            score=0.25,
            failure_type="retrieval",
            notes="unrelated policy page",
            reviewer="josh",
            review_item=item,
        )

        progress = session.get_review_queue_progress(queue.review_queue_id)
        assert progress[review_schema.ReviewItemState.COMPLETED.value] == 1

        reviews = session.get_human_reviews(
            review_queue_id=queue.review_queue_id
        )
        assert len(reviews) == 1
        assert "verdict" in reviews.to_csv(index=False)
        assert "verdict" in reviews.to_json(orient="records")

    def test_add_manually_selected_rows_to_a_queue(self, session):
        from trulens.dashboard.utils import review_utils

        queue = session.create_review_queue(name="manual")

        targets = [
            review_schema.ReviewTarget(
                target_id="r2",
                selection=review_schema.SelectionSnapshot(
                    selection_reason="manually selected", priority=0.0
                ),
            )
        ]
        session.add_review_targets(queue.review_queue_id, targets)
        session.add_review_targets(queue.review_queue_id, targets)

        # Adding twice cannot create duplicate work.
        assert (
            session.get_review_queue_progress(queue.review_queue_id)["total"]
            == 1
        )
        assert review_utils.NEW_QUEUE_LABEL


class TestPageRendering:
    """The page renders without raising for the states a reviewer hits."""

    def _run(self, session, body):
        with patch(
            "trulens.dashboard.utils.dashboard_utils.get_session",
            return_value=session,
        ), patch(
            "trulens.dashboard.tabs.Review.get_session", return_value=session
        ):
            app = AppTest.from_function(body)
            app.run(timeout=30)
            return app

    def test_queue_creation_panel_renders(self, session):
        def body():
            import pandas as pd
            from trulens.dashboard.tabs import Review

            Review.render_queue_creation(
                pd.DataFrame([
                    {
                        "record_id": "r1",
                        "Groundedness": 0.2,
                        "Groundedness direction": True,
                        "latency": 2.0,
                        "total_cost": 0.01,
                        "cost_currency": "USD",
                    }
                ])
            )

        app = self._run(session, body)
        assert not app.exception

    def test_queue_creation_panel_with_no_records(self, session):
        def body():
            import pandas as pd
            from trulens.dashboard.tabs import Review

            Review.render_queue_creation(pd.DataFrame())

        app = self._run(session, body)
        assert not app.exception
        assert any("No records" in i.value for i in app.info)

    def test_export_panel_with_no_reviews(self, session):
        def body():
            from trulens.dashboard.tabs import Review

            Review.render_export(None)

        app = self._run(session, body)
        assert not app.exception

    def test_review_panel_with_an_empty_queue(self, session):
        queue = session.create_review_queue(name="empty")

        def body():
            import streamlit as st
            from trulens.dashboard.tabs import Review

            Review.render_review_panel(st.session_state["queue_id"])

        with patch(
            "trulens.dashboard.tabs.Review.get_session", return_value=session
        ):
            app = AppTest.from_function(body)
            app.session_state["queue_id"] = queue.review_queue_id
            app.run(timeout=30)

        assert not app.exception
