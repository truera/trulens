"""Regression tests for status filtering in get_feedback.

`_feedback_query` accepts either a single
[FeedbackResultStatus][trulens.core.schema.feedback.FeedbackResultStatus] or a
sequence of them. The single-enum path used to wrap it as `[status.value]` (a
list of `str`) and then call `.value` on each element, raising `AttributeError`.
These cover both shapes on the non-OTEL query path that `get_feedback` uses.
"""

import os
import tempfile

import pandas as pd
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


def test_get_feedback_accepts_a_single_status(db, monkeypatch):
    # get_feedback is only supported with OTEL tracing disabled; exercise that
    # path, where a single status used to raise AttributeError.
    monkeypatch.setattr(db_sqlalchemy, "is_otel_tracing_enabled", lambda: False)

    result = db.get_feedback(status=FeedbackResultStatus.RUNNING)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_get_feedback_accepts_a_status_sequence(db, monkeypatch):
    monkeypatch.setattr(db_sqlalchemy, "is_otel_tracing_enabled", lambda: False)

    result = db.get_feedback(
        status=[FeedbackResultStatus.RUNNING, FeedbackResultStatus.DONE]
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
