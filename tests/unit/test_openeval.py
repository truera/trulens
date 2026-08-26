"""Tests for trulens.connectors.openeval.

These exercise to_openeval()/from_openeval() against DataFrames shaped
exactly like Run.get_records() actually returns them (record_id, input,
output, latency, plus feedback-score columns) rather than a live TruSession
-- Run itself requires a live RunDaoBase/TruSession/app just to construct,
so a fake here would be testing our own assumptions rather than TruLens's
real contract. The DataFrame *shape* is what's real: it's read directly out
of trulens.core.run.Run.get_records()'s implementation (the fixed
["record_id", "input", "output", "latency"] + metrics_columns list).

Validated against the real openeval.validate.validate_result_set()/
validate_suite(), the same real EvalPort validator every other adapter in
this ecosystem tests against -- not a mock.
"""

from openeval.validate import validate_result_set
from openeval.validate import validate_suite
import pandas as pd
import pytest
from trulens.connectors.openeval import from_openeval
from trulens.connectors.openeval import to_openeval


def _records_df(rows, attrs=None):
    df = pd.DataFrame(rows)
    if attrs:
        df.attrs.update(attrs)
    return df


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


def test_to_openeval_basic_shape_and_validates_against_real_spec():
    df = _records_df(
        [
            {
                "record_id": "rec_1",
                "input": "What is the capital of France?",
                "output": "Paris",
                "latency": 1.5,
                "Context Relevance": 0.9,
                "Groundedness": 0.8,
            },
            {
                "record_id": "rec_2",
                "input": "What is 2+2?",
                "output": "5",
                "latency": 0.4,
                "Context Relevance": 0.95,
                "Groundedness": 0.1,
            },
        ],
        attrs={"run_name": "run_20260813"},
    )

    result_set = to_openeval(df, suite_id="trulens_suite")

    assert result_set["suite_id"] == "trulens_suite"
    assert result_set["run_id"] == "run_20260813"
    assert result_set["version"]
    assert "started_at" in result_set and result_set["started_at"]
    assert "completed_at" in result_set
    assert len(result_set["results"]) == 2

    r1 = result_set["results"][0]
    assert r1["test_case_id"] == "rec_1"
    assert r1["actual_output"] == "Paris"
    assert r1["duration_ms"] == 1500
    grader_ids = {g["grader_id"] for g in r1["grader_results"]}
    assert grader_ids == {"context_relevance", "groundedness"}
    assert all(g["type"] == "custom" for g in r1["grader_results"])
    # both scores >= default 0.5 threshold -> overall passed
    assert r1["passed"] is True

    r2 = result_set["results"][1]
    # Groundedness 0.1 < default threshold 0.5 -> that grader fails,
    # and therefore the whole result fails (every grader must pass).
    assert r2["passed"] is False

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_excludes_calls_companion_columns_by_default():
    df = _records_df([
        {
            "record_id": "rec_1",
            "input": "hi",
            "output": "hello",
            "latency": 0.1,
            "Groundedness": 0.7,
            "Groundedness_calls": [{"args": {}, "ret": 0.7}],
        }
    ])
    result_set = to_openeval(df)
    grader_ids = {
        g["grader_id"] for g in result_set["results"][0]["grader_results"]
    }
    assert grader_ids == {"groundedness"}
    assert "groundedness_calls" not in grader_ids


def test_to_openeval_explicit_metric_columns_overrides_auto_detection():
    df = _records_df([
        {
            "record_id": "rec_1",
            "input": "hi",
            "output": "hello",
            "Groundedness": 0.7,
            "Custom Score": 0.3,
        }
    ])
    result_set = to_openeval(df, metric_columns=["Custom Score"])
    grader_ids = {
        g["grader_id"] for g in result_set["results"][0]["grader_results"]
    }
    assert grader_ids == {"custom_score"}


def test_to_openeval_missing_feedback_score_is_skipped_not_errored():
    df = _records_df([
        {
            "record_id": "rec_1",
            "input": "hi",
            "output": "hello",
            "Groundedness": None,
        }
    ])
    result_set = to_openeval(df)
    # No valid score -> no grader results -> overall fails cleanly, matches
    # the "no feedback scores fails cleanly" convention every other adapter
    # in the ecosystem follows (see opik-openeval-adapter's equivalent test).
    assert result_set["results"][0]["grader_results"] == []
    assert result_set["results"][0]["passed"] is False


def test_to_openeval_custom_pass_threshold():
    df = _records_df([
        {"record_id": "rec_1", "input": "x", "output": "y", "Score": 0.6}
    ])
    default = to_openeval(df)
    assert default["results"][0]["passed"] is True  # 0.6 >= default 0.5

    strict = to_openeval(df, pass_threshold=0.9)
    assert strict["results"][0]["passed"] is False  # 0.6 < 0.9


def test_to_openeval_score_clamped_to_valid_range():
    # EvalPort's schema requires score in [0, 1]; a feedback function that
    # (unusually) returns slightly out of range must not produce an invalid
    # ResultSet.
    df = _records_df([
        {"record_id": "rec_1", "input": "x", "output": "y", "Score": 1.2}
    ])
    result_set = to_openeval(df)
    assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
    assert validate_result_set(result_set).valid


def test_to_openeval_latency_unit_ms_is_respected():
    df = _records_df([
        {"record_id": "rec_1", "input": "x", "output": "y", "latency": 250}
    ])
    result_set = to_openeval(df, latency_unit="ms")
    assert result_set["results"][0]["duration_ms"] == 250


def test_to_openeval_run_id_defaults_when_no_run_name_attr():
    df = _records_df([{"record_id": "rec_1", "input": "x", "output": "y"}])
    result_set = to_openeval(df)
    assert result_set["run_id"] == "trulens_run"


def test_to_openeval_explicit_timestamps_are_respected():
    df = _records_df([{"record_id": "rec_1", "input": "x", "output": "y"}])
    result_set = to_openeval(
        df,
        started_at="2026-01-15T10:30:00Z",
        completed_at="2026-01-15T10:31:00Z",
    )
    assert result_set["started_at"] == "2026-01-15T10:30:00Z"
    assert result_set["completed_at"] == "2026-01-15T10:31:00Z"


def test_to_openeval_empty_dataframe_raises():
    with pytest.raises(ValueError):
        to_openeval(pd.DataFrame())


def test_to_openeval_missing_required_column_raises():
    df = pd.DataFrame([{"record_id": "rec_1", "input": "x"}])  # no 'output'
    with pytest.raises(ValueError, match="output"):
        to_openeval(df)


def test_to_openeval_summary_matches_actual_pass_fail_counts():
    df = _records_df([
        {"record_id": "rec_1", "input": "a", "output": "b", "Score": 0.9},
        {"record_id": "rec_2", "input": "a", "output": "b", "Score": 0.1},
        {"record_id": "rec_3", "input": "a", "output": "b", "Score": 0.6},
    ])
    result_set = to_openeval(df)
    assert result_set["summary"]["total"] == 3
    assert result_set["summary"]["passed"] == 2
    assert result_set["summary"]["failed"] == 1
    assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


def test_from_openeval_returns_dataframe_and_valid_dataset_spec():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {
                "id": "tc1",
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "graders": ["g1"],
            },
            {
                "id": "tc2",
                "input": "What is 2+2?",
                "expected_output": "4",
                "graders": ["g1"],
            },
        ],
    }

    input_df, dataset_spec = from_openeval(suite)

    assert list(input_df.columns) == [
        "input",
        "ground_truth_output",
        "input_id",
    ]
    assert len(input_df) == 2
    assert input_df.iloc[0]["input"] == "What is the capital of France?"
    assert input_df.iloc[0]["ground_truth_output"] == "Paris"
    assert input_df.iloc[0]["input_id"] == "tc1"

    assert dataset_spec == {
        "input": "input",
        "ground_truth_output": "ground_truth_output",
        "input_id": "input_id",
    }

    # dataset_spec must actually be accepted by TruLens's own validator --
    # this is the real contract RunConfig(dataset_spec=...) enforces.
    from trulens.core.run import validate_dataset_spec

    validated = validate_dataset_spec(dataset_spec)
    assert validated  # normalizes but does not reject any of our keys


def test_from_openeval_custom_input_id_column():
    suite = {
        "test_cases": [{"id": "tc1", "input": "hi", "expected_output": "hello"}]
    }
    input_df, dataset_spec = from_openeval(suite, input_id_column="my_id")
    assert "my_id" in input_df.columns
    assert dataset_spec["input_id"] == "my_id"


def test_from_openeval_missing_expected_output_becomes_none():
    suite = {"test_cases": [{"id": "tc1", "input": "hi"}]}
    input_df, _ = from_openeval(suite)
    assert input_df.iloc[0]["ground_truth_output"] is None


def test_from_openeval_empty_suite_raises():
    with pytest.raises(ValueError):
        from_openeval({"test_cases": []})


# ---------------------------------------------------------------------------
# End-to-end: EvalPort suite -> TruLens input_df -> (simulated run) ->
# TruLens records -> EvalPort ResultSet, both validated against the real spec
# ---------------------------------------------------------------------------


def test_end_to_end_suite_to_run_to_resultset_round_trip():
    suite = {
        "version": "1.0.0",
        "id": "e2e_suite",
        "graders": [
            {
                "id": "g1",
                "type": "llm_judge",
                "params": {"prompt": "{output}", "model": "gpt-4o"},
            }
        ],
        "test_cases": [
            {
                "id": "tc1",
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "graders": ["g1"],
            },
            {
                "id": "tc2",
                "input": "What is 2+2?",
                "expected_output": "4",
                "graders": ["g1"],
            },
        ],
    }
    assert validate_suite(suite).valid

    input_df, dataset_spec = from_openeval(suite)
    assert dataset_spec  # would be handed to RunConfig(dataset_spec=...)

    # Simulate what Run.get_records() would return after invocation +
    # feedback computation, using this run's own input_ids as record_ids
    # (a real integration would join TruLens's assigned record_ids back to
    # input_df[input_id_column] itself; that join is outside this module's
    # scope, same as every other adapter leaves execution to the caller).
    records_df = pd.DataFrame({
        "record_id": input_df["input_id"],
        "input": input_df["input"],
        "output": ["Paris", "4"],
        "latency": [1.1, 0.3],
        "Correctness": [1.0, 1.0],
    })
    records_df.attrs["run_name"] = "e2e_run"

    result_set = to_openeval(records_df, suite_id=suite["id"])
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert result_set["summary"]["pass_rate"] == 1.0

    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    assert all(r["test_case_id"] in suite_ids for r in result_set["results"])
