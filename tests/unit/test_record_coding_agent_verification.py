"""Tests for per-record coding-agent verification statistics.

Covers steps 5-6 of truera/trulens#2879: the `verification` and
`test_file_edits` record columns derived from the `CODING_AGENT`
attributes the client hooks record on TOOL spans, and the Leaderboard
"Verified %" aggregate computed from them.
"""

from datetime import datetime
from datetime import timedelta

from trulens.core import session as core_session
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

APP = {
    "ai.observability.app_name": "coding-agent-verification",
    "ai.observability.app_version": "v1",
    "ai.observability.app_id": "app-coding-agent-verification",
}


def _span(record_id, index, span_type, attributes):
    start = datetime(2026, 1, 1) + timedelta(seconds=index)
    return event_schema.Event.model_validate({
        "event_id": f"event-{record_id}-{index}",
        "record": {
            "kind": 1,
            "name": f"span-{index}",
            "parent_span_id": "",
            "status": "STATUS_CODE_UNSET",
        },
        "record_attributes": {
            **APP,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.SPAN_TYPE: span_type.value,
            **attributes,
        },
        "record_type": "SPAN",
        "resource_attributes": APP,
        "start_timestamp": start,
        "timestamp": start + timedelta(milliseconds=100),
        "trace": {
            "parent_id": "",
            "span_id": f"span-{record_id}-{index}",
            "trace_id": f"trace-{record_id}",
        },
    })


def _root(record_id):
    return _span(
        record_id,
        0,
        SpanAttributes.SpanType.RECORD_ROOT,
        {
            SpanAttributes.RECORD_ROOT.INPUT: "input",
            SpanAttributes.RECORD_ROOT.OUTPUT: "output",
        },
    )


def _edit(record_id, index, is_test_file):
    return _span(
        record_id,
        index,
        SpanAttributes.SpanType.TOOL,
        {
            GenAIAttributes.TOOL.NAME: "edit",
            SpanAttributes.CODING_AGENT.IS_TEST_FILE: is_test_file,
        },
    )


def _shell(record_id, index, command_kind, exit_code):
    attributes = {
        GenAIAttributes.TOOL.NAME: "bash",
        SpanAttributes.CODING_AGENT.COMMAND_KIND: command_kind,
    }
    if exit_code is not None:
        attributes[SpanAttributes.CODING_AGENT.EXIT_CODE] = exit_code
    return _span(record_id, index, SpanAttributes.SpanType.TOOL, attributes)


class TestRecordCodingAgentVerification(OtelTestCase):
    def _records(self, events):
        db = core_session.TruSession().connector.db
        self.db = db
        for event in events:
            db.insert_event(event)
        records, _ = db.get_records_and_feedback()
        return records.set_index("record_id")

    def test_edit_then_passing_test_is_passed(self):
        record_id = "edit-then-passing-test"
        row = self._records([
            _root(record_id),
            _edit(record_id, 1, False),
            _shell(record_id, 2, "test", 0),
        ]).loc[record_id]

        self.assertEqual(row["verification"], "passed")
        self.assertEqual(row["test_file_edits"], 0)

    def test_edit_then_failing_test_or_build_is_failed(self):
        failing_test = "edit-then-failing-test"
        failing_build = "edit-then-failing-build"
        records = self._records([
            _root(failing_test),
            _edit(failing_test, 1, False),
            _shell(failing_test, 2, "test", 1),
            _root(failing_build),
            _edit(failing_build, 1, False),
            _shell(failing_build, 2, "build", 2),
        ])

        self.assertEqual(records.loc[failing_test]["verification"], "failed")
        self.assertEqual(records.loc[failing_build]["verification"], "failed")

    def test_last_run_after_last_edit_decides(self):
        record_id = "fail-then-pass"
        row = self._records([
            _root(record_id),
            _edit(record_id, 1, False),
            _shell(record_id, 2, "test", 1),
            _shell(record_id, 3, "test", 0),
        ]).loc[record_id]

        self.assertEqual(row["verification"], "passed")

    def test_test_before_last_edit_does_not_count(self):
        record_id = "test-then-edit"
        row = self._records([
            _root(record_id),
            _shell(record_id, 1, "test", 0),
            _edit(record_id, 2, False),
        ]).loc[record_id]

        self.assertEqual(row["verification"], "none")

    def test_no_verification_runs_is_none_and_no_edits_is_null(self):
        no_runs = "edit-without-runs"
        no_edits = "no-edits"
        records = self._records([
            _root(no_runs),
            _edit(no_runs, 1, True),
            # Lint and VCS runs are not verification runs.
            _shell(no_runs, 2, "lint", 0),
            _shell(no_runs, 3, "vcs", 0),
            _root(no_edits),
            _shell(no_edits, 1, "test", 0),
        ])

        self.assertEqual(records.loc[no_runs]["verification"], "none")
        self.assertIsNone(records.loc[no_edits]["verification"])
        self.assertEqual(records.loc[no_edits]["test_file_edits"], 0)

    def test_test_file_edits_counts_only_test_file_edits(self):
        record_id = "mixed-edits"
        row = self._records([
            _root(record_id),
            _edit(record_id, 1, True),
            _edit(record_id, 2, False),
            _edit(record_id, 3, True),
            _shell(record_id, 4, "test", 0),
        ]).loc[record_id]

        self.assertEqual(row["test_file_edits"], 2)
        self.assertEqual(row["verification"], "passed")

    def test_string_exit_code_from_connector_is_interpreted(self):
        record_id = "string-exit-code"
        row = self._records([
            _root(record_id),
            _edit(record_id, 1, False),
            _shell(record_id, 2, "test", "0"),
        ]).loc[record_id]

        self.assertEqual(row["verification"], "passed")

    def test_leaderboard_verified_percent(self):
        events = []
        # One passed, one failed, one unverified record with edits, and
        # one record without edits: 1 passed out of 3 records with edits.
        events += [
            _root("lb-passed"),
            _edit("lb-passed", 1, False),
            _shell("lb-passed", 2, "test", 0),
        ]
        events += [
            _root("lb-failed"),
            _edit("lb-failed", 1, True),
            _shell("lb-failed", 2, "build", 1),
        ]
        events += [
            _root("lb-none"),
            _edit("lb-none", 1, False),
        ]
        events += [
            _root("lb-no-edits"),
            _shell("lb-no-edits", 1, "test", 0),
        ]
        self._records(events)

        leaderboard, feedback_columns = self.db.get_leaderboard_aggregates()
        self.assertEqual(feedback_columns, [])
        self.assertEqual(leaderboard["Records"].iloc[0], 4)
        self.assertAlmostEqual(
            leaderboard["Verified %"].iloc[0], 100.0 / 3, places=2
        )
