"""Tests that DBConnector batch-loop error logs render the exception text."""

import logging
import queue
from unittest import mock

import pytest
from trulens.core.database.connector.base import DBConnector


class _StopLoop(Exception):
    pass


class _Connector(DBConnector):
    db = mock.MagicMock()

    @classmethod
    def connect(cls, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _run_loop_once(connector):
    # The loop sleeps first on every iteration; let the first sleep pass and
    # abort on the second one.
    with mock.patch(
        "trulens.core.database.connector.base.time.sleep",
        side_effect=[None, _StopLoop()],
    ):
        with pytest.raises(_StopLoop):
            connector._batch_loop()


def test_batch_insert_record_failure_logs_exception_text(caplog):
    connector = object.__new__(_Connector)
    connector.batch_record_queue = queue.Queue()
    connector.batch_record_queue.put(mock.MagicMock())
    failing_db = mock.MagicMock()
    failing_db.batch_insert_record.side_effect = RuntimeError("boom-record")
    with mock.patch.object(_Connector, "db", failing_db):
        with caplog.at_level(logging.ERROR):
            _run_loop_once(connector)

    assert "boom-record" in caplog.text
    assert "Logging error" not in caplog.text


def test_batch_insert_feedback_failure_logs_exception_text(caplog):
    connector = object.__new__(_Connector)
    connector.batch_record_queue = queue.Queue()
    record = mock.MagicMock()
    connector.batch_record_queue.put(record)
    db = mock.MagicMock()
    db.batch_insert_feedback.side_effect = RuntimeError("boom-feedback")
    connector.get_app = mock.MagicMock(return_value={})
    with mock.patch.object(_Connector, "db", db):
        with caplog.at_level(logging.ERROR):
            _run_loop_once(connector)

    assert "boom-feedback" in caplog.text
