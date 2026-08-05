import datetime

import pandas as pd
from trulens.dashboard.tabs.Trends import _filter_summary
from trulens.dashboard.tabs.Trends import _render_trend_chart
from trulens.dashboard.tabs.Trends import _version_selector_data

from tests.unit.streamlit.test_streamlit_utils import AppTestHelper


def test_version_selector_defaults_most_active_version_selected():
    versions = pd.DataFrame({"app_version": ["v1", "v2"]})
    aggregates = pd.DataFrame({
        "app_version": ["v1", "v2"],
        "Records": [10, 20],
        "Recent Records": [8, 2],
        "Latest Record Timestamp": pd.to_datetime(["2026-08-05", "2026-08-04"]),
        "Sample Rate": [1.0, 0.1],
        "Sample Rate Min": [1.0, 0.05],
        "Sample Rate Max": [1.0, 0.1],
    })

    result = _version_selector_data(versions, aggregates)

    assert result["Display"].tolist() == [True, False]
    assert result["Sample rate"].tolist() == ["100%", "10%"]


def test_filter_summary_is_compact():
    assert (
        _filter_summary(
            ["v3"],
            datetime.date(2026, 7, 1),
            datetime.date(2026, 7, 30),
            "day",
        )
        == "v3 · Jul 1–30, 2026 · Day"
    )
    assert (
        _filter_summary(
            ["v2", "v3"],
            datetime.date(2026, 7, 30),
            datetime.date(2026, 8, 5),
            "week",
        )
        == "2 versions · Jul 30, 2026–Aug 5, 2026 · Week"
    )


def test_combined_chart_renders_every_metric_version_series():
    def test_app():
        import pandas as pd

        trends = pd.DataFrame({
            "app_version": ["v1", "v2", "v1", "v2"],
            "metric_name": [
                "groundedness",
                "groundedness",
                "relevance",
                "relevance",
            ],
            "time_bucket": pd.to_datetime(["2026-07-01"] * 4),
            "count": [10] * 4,
            "mean": [0.6, 0.7, 0.8, 0.9],
            "ci_lower": [0.5, 0.6, 0.7, 0.8],
            "ci_upper": [0.7, 0.8, 0.9, 1.0],
        })
        _render_trend_chart(
            trends, "day", {"groundedness": True, "relevance": True}
        )

    app = AppTestHelper.create_and_run_app(test_app)

    AppTestHelper.assert_no_errors(app)


def test_latency_and_cost_charts_render():
    def test_app():
        import pandas as pd
        from trulens.dashboard.tabs.Trends import _render_cost_chart
        from trulens.dashboard.tabs.Trends import _render_latency_chart

        metrics = pd.DataFrame({
            "app_version": ["v1"],
            "time_bucket": pd.to_datetime(["2026-07-01"]),
            "currency": ["USD"],
            "average_latency": [0.8],
            "p90_latency": [1.1],
            "p99_latency": [1.4],
            "total_app_cost": [0.1],
        })
        _render_latency_chart(metrics, "day")
        _render_cost_chart(
            metrics, "total_app_cost", "Total app cost", "app_cost", "day"
        )

    app = AppTestHelper.create_and_run_app(test_app)

    AppTestHelper.assert_no_errors(app)
