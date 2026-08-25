"""Leaderboard metric coverage: the mean and `Records` have different
denominators, so a metric that did not score every record must say so."""

import pandas as pd
import pytest
from trulens.dashboard.tabs.Leaderboard import _preprocess_df
from trulens.dashboard.tabs.Leaderboard import scored_count_col

from tests.unit.streamlit.test_streamlit_utils import AppTestHelper

METRIC = "Correctness"


def _records(scores) -> pd.DataFrame:
    n = len(scores)
    return pd.DataFrame({
        "app_id": ["app"] * n,
        "app_name": ["Test App"] * n,
        "app_version": ["v1"] * n,
        "record_id": [f"record-{i}" for i in range(n)],
        "latency": [1.0] * n,
        "total_cost": [0.0] * n,
        "cost_currency": ["USD"] * n,
        "total_tokens": [1] * n,
        "tags": [""] * n,
        METRIC: scores,
    })


def _app_versions() -> pd.DataFrame:
    return pd.DataFrame({
        "app_id": ["app"],
        "app_name": ["Test App"],
        "app_version": ["v1"],
    })


def _leaderboard(scores) -> pd.Series:
    df = _preprocess_df(_records(scores), _app_versions(), [METRIC], [])
    return df.iloc[0]


def test_scored_count_matches_records_when_every_record_is_scored():
    row = _leaderboard([0.1, 0.2, 0.3, 0.4, 0.5])

    assert row["Records"] == 5
    assert row[scored_count_col(METRIC)] == 5


def test_scored_count_excludes_records_whose_metric_did_not_produce_a_score():
    # A metric that raised is stored with status FAILED and no result, so its
    # column is NaN. `Records` still counts those records.
    row = _leaderboard([None, None, 0.8, 0.9, 1.0])

    assert row["Records"] == 5
    assert row[scored_count_col(METRIC)] == 3


def test_mean_over_a_subset_is_reported_with_its_own_denominator():
    # Same five records, two ways: fully scored, and with the two lowest
    # scores missing. The mean rises because the low scores are gone -- the
    # scored count is what makes that visible.
    full = _leaderboard([0.1, 0.2, 0.8, 0.9, 1.0])
    partial = _leaderboard([None, None, 0.8, 0.9, 1.0])

    assert partial[METRIC] > full[METRIC]
    assert partial["Records"] == full["Records"] == 5
    assert partial[scored_count_col(METRIC)] == 3
    assert full[scored_count_col(METRIC)] == 5


def _captions(app) -> str:
    return " | ".join(str(caption.value) for caption in app.caption)


def test_partially_scored_metric_shows_its_coverage():
    def test_app():
        import pandas as pd
        from trulens.dashboard.tabs.Leaderboard import _preprocess_df
        from trulens.dashboard.tabs.Leaderboard import _render_list_tab

        scores = [None, None, 0.8, 0.9, 1.0]
        records = pd.DataFrame({
            "app_id": ["app"] * 5,
            "app_name": ["Test App"] * 5,
            "app_version": ["v1"] * 5,
            "record_id": [f"record-{i}" for i in range(5)],
            "latency": [1.0] * 5,
            "total_cost": [0.0] * 5,
            "cost_currency": ["USD"] * 5,
            "total_tokens": [1] * 5,
            "tags": [""] * 5,
            "Correctness": scores,
        })
        versions = pd.DataFrame({
            "app_id": ["app"],
            "app_name": ["Test App"],
            "app_version": ["v1"],
        })
        df = _preprocess_df(records, versions, ["Correctness"], [])
        _render_list_tab(df, ["Correctness"], {"Correctness": True}, [])

    app = AppTestHelper.create_and_run_app(test_app)
    AppTestHelper.assert_no_errors(app)
    assert "Scored 3 of 5 records" in _captions(app)


def test_fully_scored_metric_shows_no_coverage_caption():
    def test_app():
        import pandas as pd
        from trulens.dashboard.tabs.Leaderboard import _preprocess_df
        from trulens.dashboard.tabs.Leaderboard import _render_list_tab

        scores = [0.1, 0.2, 0.8, 0.9, 1.0]
        records = pd.DataFrame({
            "app_id": ["app"] * 5,
            "app_name": ["Test App"] * 5,
            "app_version": ["v1"] * 5,
            "record_id": [f"record-{i}" for i in range(5)],
            "latency": [1.0] * 5,
            "total_cost": [0.0] * 5,
            "cost_currency": ["USD"] * 5,
            "total_tokens": [1] * 5,
            "tags": [""] * 5,
            "Correctness": scores,
        })
        versions = pd.DataFrame({
            "app_id": ["app"],
            "app_name": ["Test App"],
            "app_version": ["v1"],
        })
        df = _preprocess_df(records, versions, ["Correctness"], [])
        _render_list_tab(df, ["Correctness"], {"Correctness": True}, [])

    app = AppTestHelper.create_and_run_app(test_app)
    AppTestHelper.assert_no_errors(app)
    assert "Scored" not in _captions(app)


def test_scored_count_column_is_hidden_in_the_grid():
    pytest.importorskip("st_aggrid")
    from trulens.dashboard.tabs.Leaderboard import _build_grid_options

    df = _preprocess_df(
        _records([None, None, 0.8, 0.9, 1.0]),
        _app_versions(),
        [METRIC],
        [],
    )
    options = _build_grid_options(
        df=df,
        feedback_col_names=[METRIC],
        feedback_directions={METRIC: True},
        version_metadata_col_names=[],
    )

    visible = [
        c.get("field") for c in options["columnDefs"] if not c.get("hide")
    ]
    assert METRIC in visible
    assert scored_count_col(METRIC) not in visible
