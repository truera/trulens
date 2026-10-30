"""Feedback trends exclude judge failures from score statistics."""

import math

import pandas as pd
import pytest
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.schema import event as event_schema
from trulens.core.utils import constants as constants_utils
from trulens.otel.semconv import trace as semconv_trace


@pytest.fixture
def db(tmp_path):
    database = db_sqlalchemy.SQLAlchemyDB.from_db_url(
        f"sqlite:///{tmp_path / 'trends.sqlite'}"
    )
    database.migrate_database()
    yield database
    database.engine.dispose()


def _seed_scores(db, scores):
    """Persist real eval-root events in one metric and time bucket."""
    attributes = semconv_trace.SpanAttributes
    resources = semconv_trace.ResourceAttributes
    for index, score in enumerate(scores):
        db.insert_event(
            event_schema.Event.model_validate({
                "event_id": f"eval-{index}",
                "record": {"name": "relevance", "kind": 1},
                "record_attributes": {
                    attributes.SPAN_TYPE: attributes.SpanType.EVAL_ROOT.value,
                    attributes.RECORD_ID: f"record-{index}",
                    attributes.EVAL_ROOT.METRIC_NAME: "relevance",
                    attributes.EVAL_ROOT.SCORE: score,
                },
                "record_type": "SPAN",
                "resource_attributes": {
                    resources.APP_NAME: "trends",
                    resources.APP_VERSION: "v1",
                },
                "start_timestamp": "2026-01-01T00:00:00",
                "timestamp": "2026-01-01T00:00:01",
                "trace": {
                    "parent_id": "",
                    "span_id": f"span-{index}",
                    "trace_id": "trace",
                },
            })
        )


@pytest.mark.parametrize(
    "scores, mean, standard_error",
    [
        ([1.0, 1.0, constants_utils.UNPARSABLE_SCORE], 1.0, 0.0),
        ([0.4, 0.6, constants_utils.UNPARSABLE_SCORE], 0.5, 0.1),
        ([0.4, 0.6], 0.5, 0.1),
        ([-0.5, 0.5, constants_utils.UNPARSABLE_SCORE], 0.0, 0.5),
    ],
)
def test_trends_use_only_graded_scores(db, scores, mean, standard_error):
    _seed_scores(db, scores)

    trends = db.get_feedback_score_trends(app_name="trends")

    assert len(trends) == 1
    row = trends.iloc[0]
    assert row["count"] == 2
    assert row["mean"] == pytest.approx(mean)
    assert row["standard_error"] == pytest.approx(standard_error)
    assert row["ci_lower"] == pytest.approx(
        max(0.0, mean - 1.96 * standard_error)
    )
    assert row["ci_upper"] == pytest.approx(
        min(1.0, mean + 1.96 * standard_error)
    )


def test_ungraded_bucket_has_no_measured_trend(db):
    _seed_scores(db, [constants_utils.UNPARSABLE_SCORE] * 2)

    assert db.get_feedback_score_trends().empty


def test_single_graded_score_has_no_confidence_interval(db):
    _seed_scores(db, [0.8, constants_utils.UNPARSABLE_SCORE])

    row = db.get_feedback_score_trends().iloc[0]

    assert row["count"] == 1
    assert row["mean"] == pytest.approx(0.8)
    assert math.isnan(row["standard_error"])
    assert pd.isna(row["ci_lower"])
    assert pd.isna(row["ci_upper"])
