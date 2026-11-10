from datetime import datetime
from datetime import timezone
from typing import Generator

import pandas as pd
import pytest
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.schema.event import Event
from trulens.core.schema.event import EventRecordType
from trulens.core.utils import constants as constants_utils
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> Generator[SQLAlchemyDB, None, None]:
    monkeypatch.setenv("TRULENS_OTEL_TRACING", "1")
    database = SQLAlchemyDB.from_db_url("sqlite:///:memory:")
    assert database.engine is not None
    engine = database.engine

    try:
        database.migrate_database()
        yield database
    finally:
        engine.dispose()


def _insert_scores(database: SQLAlchemyDB, scores: list[float]) -> None:
    timestamp = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    events = [
        Event(
            event_id=f"event-{index}",
            record={"name": "evaluation"},
            record_attributes={
                SpanAttributes.SPAN_TYPE: (
                    SpanAttributes.SpanType.EVAL_ROOT.value
                ),
                SpanAttributes.RECORD_ID: f"record-{index}",
                SpanAttributes.EVAL_ROOT.METRIC_NAME: "answer_relevance",
                SpanAttributes.EVAL_ROOT.SCORE: score,
            },
            record_type=EventRecordType.SPAN,
            resource_attributes={
                ResourceAttributes.APP_NAME: "test-app",
                ResourceAttributes.APP_VERSION: "v1",
            },
            start_timestamp=timestamp,
            timestamp=timestamp,
            trace={
                "trace_id": f"trace-{index}",
                "parent_id": "",
                "span_id": f"span-{index}",
            },
        )
        for index, score in enumerate(scores)
    ]
    database.insert_events(events)


@pytest.mark.parametrize(
    (
        "scores",
        "expected_count",
        "expected_mean",
        "expected_standard_error",
        "expected_ci_lower",
        "expected_ci_upper",
    ),
    [
        ([1.0, 1.0, constants_utils.UNPARSABLE_SCORE], 2, 1.0, 0.0, 1.0, 1.0),
        (
            [0.4, 0.6, constants_utils.UNPARSABLE_SCORE],
            2,
            0.5,
            0.1,
            0.304,
            0.696,
        ),
        ([-0.5, 0.5, constants_utils.UNPARSABLE_SCORE], 2, 0.0, 0.5, 0.0, 0.98),
        (
            [0.0, 0.5, constants_utils.UNPARSABLE_SCORE],
            2,
            0.25,
            0.25,
            0.0,
            0.74,
        ),
    ],
)
def test_feedback_score_trends_excludes_only_unparsable_sentinel(
    db: SQLAlchemyDB,
    scores: list[float],
    expected_count: int,
    expected_mean: float,
    expected_standard_error: float,
    expected_ci_lower: float,
    expected_ci_upper: float,
) -> None:
    _insert_scores(db, scores)

    trends = db.get_feedback_score_trends()

    assert len(trends) == 1
    trend = trends.iloc[0]
    assert trend["count"] == expected_count
    assert trend["mean"] == pytest.approx(expected_mean)
    assert trend["standard_error"] == pytest.approx(expected_standard_error)
    assert trend["ci_lower"] == pytest.approx(expected_ci_lower)
    assert trend["ci_upper"] == pytest.approx(expected_ci_upper)


def test_feedback_score_trends_omits_bucket_with_only_unparsable_scores(
    db: SQLAlchemyDB,
) -> None:
    _insert_scores(db, [constants_utils.UNPARSABLE_SCORE] * 2)

    trends = db.get_feedback_score_trends()

    assert trends.empty


def test_feedback_score_trends_single_graded_score_has_no_confidence_interval(
    db: SQLAlchemyDB,
) -> None:
    _insert_scores(db, [0.8, constants_utils.UNPARSABLE_SCORE])

    trends = db.get_feedback_score_trends()

    assert len(trends) == 1
    trend = trends.iloc[0]
    assert trend["count"] == 1
    assert trend["mean"] == pytest.approx(0.8)
    assert pd.isna(trend["standard_error"])
    assert pd.isna(trend["ci_lower"])
    assert pd.isna(trend["ci_upper"])


def test_feedback_score_trends_preserves_fully_graded_bucket(
    db: SQLAlchemyDB,
) -> None:
    _insert_scores(db, [0.4, 0.6])

    trends = db.get_feedback_score_trends()

    assert len(trends) == 1
    trend = trends.iloc[0]
    assert trend["count"] == 2
    assert trend["mean"] == pytest.approx(0.5)
