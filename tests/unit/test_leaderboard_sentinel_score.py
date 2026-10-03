"""Leaderboard aggregates must not average the unparsable-score sentinel.

`llm_provider.UNPARSABLE_SCORE` (-1.0) is stored as a feedback `result` when an
LLM judge returns nothing parseable. `BatchEvaluator._aggregate_scores` filters
that sentinel out before aggregating (see #2866), but both leaderboard
aggregation paths used a bare `AVG(result)`, so a single judge failure dragged
the reported average down: `[1.0, 1.0, -1.0]` surfaced as `0.33` instead of
`1.0`. That reads as a mediocre metric rather than a failed judge.

When every score for a metric is a sentinel there is nothing parsable to
average, and the sentinel itself is reported so the failure stays visible.
"""

import datetime
import os
import tempfile

import pytest
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.schema.feedback import FeedbackResultStatus


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tempdir:
        database = db_sqlalchemy.SQLAlchemyDB.from_db_url(
            f"sqlite:///{os.path.join(tempdir, 'trulens.sqlite')}"
        )
        database.migrate_database()
        yield database


def _seed_scores(db, name, scores):
    """Store one record per score under a single metric called *name*."""
    app_id = "sentinel_app_v1"
    with db.session.begin() as s:
        s.add(
            db.orm.AppDefinition(
                app_id=app_id,
                app_name="sentinel_app",
                app_version="v1",
                app_json='{"app_name": "sentinel_app", "app_version": "v1"}',
            )
        )
        s.add(
            db.orm.FeedbackDefinition(
                feedback_definition_id=f"{name}_def",
                run_location=None,
                feedback_json=f'{{"impl": {"null"}}}',
            )
        )

    now = datetime.datetime(2026, 1, 1).timestamp()
    with db.session.begin() as s:
        for i, score in enumerate(scores):
            record_id = f"{name}_record_{i}"
            s.add(
                db.orm.Record(
                    record_id=record_id,
                    app_id=app_id,
                    input=None,
                    output=None,
                    record_json="{}",
                    tags="[]",
                    ts=now,
                    cost_json="{}",
                    perf_json="{}",
                )
            )
            s.add(
                db.orm.FeedbackResult(
                    feedback_result_id=f"{name}_result_{i}",
                    record_id=record_id,
                    feedback_definition_id=f"{name}_def",
                    last_ts=now,
                    status=FeedbackResultStatus.DONE.value,
                    error=None,
                    calls_json="{}",
                    result=score,
                    name=name,
                    cost_json="{}",
                    multi_result=None,
                )
            )


def _avg_for(db, name, monkeypatch):
    """Read back the reported average for *name* from the leaderboard."""
    monkeypatch.setattr(db_sqlalchemy, "is_otel_tracing_enabled", lambda: False)
    frame, cols = db.get_leaderboard_aggregates()
    assert name in cols, f"{name} missing from leaderboard columns {cols}"
    return frame[name].iloc[0]


def test_sentinel_does_not_dilute_the_average(db, monkeypatch):
    """Two good scores plus one sentinel must average the two good ones."""
    _seed_scores(db, "relevance", [1.0, 1.0, -1.0])

    assert _avg_for(db, "relevance", monkeypatch) == 1.0


def test_all_sentinel_scores_report_the_sentinel(db, monkeypatch):
    """With nothing parsable left, report the sentinel rather than a plain 0."""
    _seed_scores(db, "relevance", [-1.0, -1.0])

    assert _avg_for(db, "relevance", monkeypatch) == -1.0


def test_sentinel_is_excluded_but_other_negatives_are_kept(db, monkeypatch):
    """Only the sentinel is filtered: -0.5 is a legitimate score to average."""
    _seed_scores(db, "sentiment", [0.5, -0.5, -1.0])

    assert _avg_for(db, "sentiment", monkeypatch) == 0.0
