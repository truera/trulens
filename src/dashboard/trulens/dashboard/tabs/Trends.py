import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from trulens.dashboard import constants as dashboard_constants
from trulens.dashboard.utils import dashboard_utils

page_name = dashboard_constants.TRENDS_PAGE_NAME

METRIC_COLORS = (
    "#29B5E8",
    "#7C3AED",
    "#10B981",
    "#F59E0B",
    "#EF4444",
    "#6366F1",
)
VERSION_DASHES = ("solid", "dash", "dot", "dashdot")


def init_page_state() -> None:
    if st.session_state.get(f"{page_name}.initialized", False):
        return
    dashboard_utils.read_query_params_into_session_state(page_name=page_name)
    st.session_state[f"{page_name}.initialized"] = True


def _version_selector_data(
    versions_df: pd.DataFrame, aggregates_df: pd.DataFrame
) -> pd.DataFrame:
    aggregates = aggregates_df.reset_index()
    rates = aggregates[
        [
            "app_version",
            "Records",
            "Recent Records",
            "Latest Record Timestamp",
            "Sample Rate",
            "Sample Rate Min",
            "Sample Rate Max",
        ]
    ]
    selector = versions_df[["app_version"]].merge(
        rates, on="app_version", how="left"
    )
    selector["Display"] = selector.apply(
        dashboard_utils.format_leaderboard_sample_rate, axis=1
    )
    pinned_col = dashboard_constants.PINNED_COL_NAME
    pinned_versions = (
        versions_df.loc[versions_df[pinned_col], "app_version"].tolist()
        if pinned_col in versions_df.columns
        else []
    )
    if pinned_versions:
        default_version = pinned_versions[0]
    else:
        default_version = selector.sort_values(
            [
                "Recent Records",
                "Latest Record Timestamp",
                "Records",
                "app_version",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]["app_version"]
    return pd.DataFrame({
        "Display": selector["app_version"] == default_version,
        "App version": selector["app_version"],
        "Sample rate": selector["Display"],
    })


def _render_controls(
    versions_df: pd.DataFrame,
    aggregates_df: pd.DataFrame,
    available_trends: pd.DataFrame,
) -> tuple[list[str], datetime.date, datetime.date, str]:
    minimum_date = available_trends["time_bucket"].min().date()
    maximum_date = available_trends["time_bucket"].max().date()
    with st.popover("Filters", icon=":material/filter_list:", width="stretch"):
        st.subheader("App versions")
        selected = st.data_editor(
            _version_selector_data(versions_df, aggregates_df),
            key=f"{page_name}.version_selector.v2",
            hide_index=True,
            disabled=["App version", "Sample rate"],
            column_config={
                "Display": st.column_config.CheckboxColumn("Display"),
            },
            width="stretch",
        )
        st.subheader("Time range")
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input(
            "Start date",
            value=minimum_date,
            min_value=minimum_date,
            max_value=maximum_date,
            key=f"{page_name}.start_date",
        )
        end_date = date_cols[1].date_input(
            "End date",
            value=maximum_date,
            min_value=minimum_date,
            max_value=maximum_date,
            key=f"{page_name}.end_date",
        )
        bucket = st.segmented_control(
            "Bucket",
            ["Day", "Week"],
            default="Day",
            key=f"{page_name}.bucket",
        )
    selected_versions = selected.loc[
        selected["Display"], "App version"
    ].tolist()
    return selected_versions, start_date, end_date, bucket.lower()


def _filter_summary(
    selected_versions: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    bucket: str,
) -> str:
    versions = (
        selected_versions[0]
        if len(selected_versions) == 1
        else f"{len(selected_versions)} versions"
    )
    date_range = (
        f"{start_date:%b %-d}–{end_date:%-d, %Y}"
        if start_date.month == end_date.month
        and start_date.year == end_date.year
        else f"{start_date:%b %-d, %Y}–{end_date:%b %-d, %Y}"
    )
    return f"{versions} · {date_range} · {bucket.title()}"


def _bucket_end(value, bucket: str) -> pd.Timestamp:
    return pd.Timestamp(value) + pd.Timedelta(days=1 if bucket == "day" else 7)


def _selection_metadata(event) -> list[list] | None:
    if event is None or not event.selection.points:
        return None
    selections = [
        point.get("customdata")
        for point in event.selection.points
        if point.get("customdata") and len(point["customdata"]) >= 7
    ]
    return selections or None


def _render_trend_chart(
    trends: pd.DataFrame,
    bucket: str,
    feedback_directions: dict[str, bool],
) -> list | None:
    metric_names = sorted(trends["metric_name"].unique())
    app_versions = sorted(trends["app_version"].unique())
    metric_colors = {
        metric: METRIC_COLORS[index % len(METRIC_COLORS)]
        for index, metric in enumerate(metric_names)
    }
    version_dashes = {
        version: VERSION_DASHES[index % len(VERSION_DASHES)]
        for index, version in enumerate(app_versions)
    }
    fig = go.Figure()
    for (metric_name, app_version), series in trends.groupby(
        ["metric_name", "app_version"], sort=True
    ):
        series = series.sort_values("time_bucket")
        color = metric_colors[metric_name]
        label = f"{metric_name} · {app_version}"
        fig.add_trace(
            go.Scatter(
                x=series["time_bucket"],
                y=series["mean"],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line={
                    "color": color,
                    "dash": version_dashes[app_version],
                },
                customdata=[
                    [
                        "feedback",
                        metric_name,
                        app_version,
                        pd.Timestamp(time_bucket).isoformat(),
                        _bucket_end(time_bucket, bucket).isoformat(),
                        "asc"
                        if feedback_directions.get(metric_name, True)
                        else "desc",
                        "",
                        count,
                    ]
                    for time_bucket, count in zip(
                        series["time_bucket"], series["count"]
                    )
                ],
                hovertemplate=(
                    "%{x}<br>Mean: %{y:.3f}<br>Records: "
                    "%{customdata[7]}<extra>%{fullData.name}</extra>"
                ),
            )
        )
        valid_ci = series.dropna(subset=["ci_lower", "ci_upper"])
        if valid_ci.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=list(valid_ci["time_bucket"])
                + list(valid_ci["time_bucket"][::-1]),
                y=list(valid_ci["ci_upper"]) + list(valid_ci["ci_lower"][::-1]),
                fill="toself",
                fillcolor=color,
                line={"color": "rgba(0,0,0,0)"},
                opacity=0.08,
                hoverinfo="skip",
                legendgroup=label,
                showlegend=False,
            )
        )
    fig.update_layout(
        height=600,
        dragmode="select",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis={"range": [0, 1]},
        legend={
            "orientation": "h",
            "y": -0.2,
            "title": {
                "text": (
                    "Color = metric · line style = app version · "
                    "shading = 95% confidence interval"
                )
            },
        },
        margin={"b": 170},
    )
    event = st.plotly_chart(
        fig,
        key=f"{page_name}.evaluation_chart",
        on_select="rerun",
        selection_mode=("points", "box"),
        width="stretch",
    )
    return _selection_metadata(event)


def _render_latency_chart(metrics: pd.DataFrame, bucket: str) -> list | None:
    statistics = {
        "Average": ("average_latency", "#29B5E8"),
        "P90": ("p90_latency", "#7C3AED"),
        "P99": ("p99_latency", "#EF4444"),
    }
    versions = sorted(metrics["app_version"].unique())
    dashes = {
        version: VERSION_DASHES[index % len(VERSION_DASHES)]
        for index, version in enumerate(versions)
    }
    fig = go.Figure()
    for version in versions:
        version_data = metrics[metrics["app_version"] == version].sort_values(
            "time_bucket"
        )
        for label, (column, color) in statistics.items():
            fig.add_trace(
                go.Scatter(
                    x=version_data["time_bucket"],
                    y=version_data[column],
                    mode="lines+markers",
                    name=f"{label} · {version}",
                    line={"color": color, "dash": dashes[version]},
                    customdata=[
                        [
                            "latency",
                            label,
                            version,
                            pd.Timestamp(ts).isoformat(),
                            _bucket_end(ts, bucket).isoformat(),
                            "desc",
                            "",
                        ]
                        for ts in version_data["time_bucket"]
                    ],
                    hovertemplate=(
                        "%{x}<br>Latency: %{y:.3f}s"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    fig.update_layout(
        height=460,
        dragmode="select",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Latency (seconds)",
        legend={"orientation": "h", "y": -0.2},
        margin={"b": 120},
    )
    event = st.plotly_chart(
        fig,
        key=f"{page_name}.latency_chart",
        on_select="rerun",
        selection_mode=("points", "box"),
        width="stretch",
    )
    return _selection_metadata(event)


def _render_cost_chart(
    costs: pd.DataFrame,
    value_column: str,
    title: str,
    metric_kind: str,
    bucket: str,
) -> list | None:
    fig = go.Figure()
    series_keys = costs[["app_version", "currency"]].drop_duplicates()
    for index, row in series_keys.reset_index(drop=True).iterrows():
        version = row["app_version"]
        currency = row["currency"]
        series = costs[
            (costs["app_version"] == version) & (costs["currency"] == currency)
        ].sort_values("time_bucket")
        fig.add_trace(
            go.Scatter(
                x=series["time_bucket"],
                y=series[value_column],
                mode="lines+markers",
                name=f"{version} · {currency}",
                line={
                    "color": METRIC_COLORS[index % len(METRIC_COLORS)],
                    "dash": VERSION_DASHES[index % len(VERSION_DASHES)],
                },
                customdata=[
                    [
                        metric_kind,
                        title,
                        version,
                        pd.Timestamp(ts).isoformat(),
                        _bucket_end(ts, bucket).isoformat(),
                        "desc",
                        currency,
                    ]
                    for ts in series["time_bucket"]
                ],
                hovertemplate=(
                    "%{x}<br>Cost: %{y:.6f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=380,
        dragmode="select",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Cost",
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 100},
    )
    event = st.plotly_chart(
        fig,
        key=f"{page_name}.{metric_kind}.{value_column}",
        on_select="rerun",
        selection_mode=("points", "box"),
        width="stretch",
    )
    return _selection_metadata(event)


def _render_investigation(
    selections: list[list] | None, chart_key: str
) -> None:
    if not selections:
        return
    kind = selections[0][0]
    metrics = sorted({selection[1] for selection in selections})
    versions = sorted({selection[2] for selection in selections})
    start_time = min(pd.Timestamp(selection[3]) for selection in selections)
    end_time = max(pd.Timestamp(selection[4]) for selection in selections)
    direction = selections[0][5]
    currencies = sorted({
        selection[6] for selection in selections if selection[6]
    })
    with st.container(border=True):
        cols = st.columns([4, 1], vertical_alignment="center")
        cols[0].markdown(
            f"**Investigate:** {len(metrics)} metric(s) · {len(versions)} "
            f"version(s) · {start_time:%b %d}–{(end_time - pd.Timedelta(days=1)):%b %d}"
        )
        cols[0].caption(
            "Records will be sorted "
            + ("lowest first." if direction == "asc" else "highest first.")
        )
        if cols[1].button(
            "Investigate records",
            key=f"{page_name}.investigate.{chart_key}",
            type="primary",
            width="stretch",
        ):
            drilldown = dashboard_utils.RecordsDrilldown(
                app_version=",".join(versions),
                metric_kind=kind,
                metric_name=",".join(metrics),
                start_time=start_time.to_pydatetime(),
                end_time=end_time.to_pydatetime(),
                sort_direction=direction,
                currency=",".join(currencies) or None,
            )
            for name, value in drilldown.to_query_params().items():
                st.session_state[f"Records.{name}"] = (
                    value.split(",")
                    if name in ("app_versions", "metric_name")
                    else value
                )
                st.query_params[name] = value
            st.session_state.pop("Records.selected_thread", None)
            st.session_state.pop("Records.selected_record", None)
            st.switch_page("tabs/Records.py")


def render_trends(app_name: str) -> None:
    versions_df, _ = dashboard_utils.get_app_versions(app_name)
    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return
    app_versions = versions_df["app_version"].tolist()
    aggregates_df, _ = dashboard_utils.get_leaderboard_aggregates(
        app_name=app_name, app_versions=app_versions
    )
    available_trends = dashboard_utils.get_feedback_score_trends(
        app_name=app_name, app_versions=app_versions, bucket="day"
    )
    if available_trends.empty:
        st.info("No score trend data available.")
        return
    header_cols = st.columns([6, 1], vertical_alignment="bottom")
    with header_cols[0]:
        st.title(page_name)
        st.markdown(f"Showing app `{app_name}`")
    with header_cols[1]:
        selected_versions, start_date, end_date, bucket = _render_controls(
            versions_df, aggregates_df, available_trends
        )
    st.caption(_filter_summary(selected_versions, start_date, end_date, bucket))
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return
    if not selected_versions:
        st.info("Select at least one app version to display trends.")
        return
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    trends = dashboard_utils.get_feedback_score_trends(
        app_name=app_name,
        app_versions=selected_versions,
        bucket=bucket,
        start_time=pd.Timestamp(start_date),
        end_time=end_exclusive,
    )
    if trends.empty:
        st.info("No score trend data available for the selected time range.")
        return
    _, feedback_directions = dashboard_utils.get_feedback_defs()
    st.subheader("Evaluation Metrics")
    evaluation_selections = _render_trend_chart(
        trends, bucket, feedback_directions
    )
    if evaluation_selections:
        selected_versions_set = {item[2] for item in evaluation_selections}
        selected_start = min(
            pd.Timestamp(item[3]) for item in evaluation_selections
        )
        selected_end = max(
            pd.Timestamp(item[4]) for item in evaluation_selections
        )
        evaluation_selections = [
            [
                "feedback",
                metric,
                version,
                selected_start.isoformat(),
                selected_end.isoformat(),
                "asc" if feedback_directions.get(metric, True) else "desc",
                "",
            ]
            for metric in sorted(trends["metric_name"].unique())
            for version in sorted(selected_versions_set)
        ]
    _render_investigation(evaluation_selections, "evaluation")
    app_metrics = dashboard_utils.get_app_metric_trends(
        app_name=app_name,
        app_versions=selected_versions,
        bucket=bucket,
        start_time=pd.Timestamp(start_date),
        end_time=end_exclusive,
    )
    st.subheader("Latency")
    if app_metrics.empty:
        st.info("No latency data available for the selected time range.")
    else:
        _render_investigation(
            _render_latency_chart(app_metrics, bucket), "latency"
        )
    st.subheader("App Cost")
    if app_metrics.empty or not (app_metrics["total_app_cost"] > 0).any():
        st.info(
            "No application cost data available for the selected time range."
        )
    else:
        cost_cols = st.columns(2)
        with cost_cols[0]:
            total_app_selection = _render_cost_chart(
                app_metrics,
                "total_app_cost",
                "Total app cost",
                "app_cost",
                bucket,
            )
        with cost_cols[1]:
            average_app_selection = _render_cost_chart(
                app_metrics,
                "average_app_cost",
                "Average app cost per record",
                "app_cost",
                bucket,
            )
        _render_investigation(
            total_app_selection or average_app_selection, "app_cost"
        )
    eval_costs = dashboard_utils.get_eval_cost_trends(
        app_name=app_name,
        app_versions=selected_versions,
        bucket=bucket,
        start_time=pd.Timestamp(start_date),
        end_time=end_exclusive,
    )
    st.subheader("Evaluation Cost")
    if eval_costs.empty:
        st.info("No evaluation cost data reported for the selected time range.")
    else:
        eval_cols = st.columns(2)
        with eval_cols[0]:
            total_eval_selection = _render_cost_chart(
                eval_costs, "total_eval_cost", "Total cost", "eval_cost", bucket
            )
        with eval_cols[1]:
            average_eval_selection = _render_cost_chart(
                eval_costs,
                "average_eval_cost",
                "Average cost per evaluation",
                "eval_cost",
                bucket,
            )
        _render_investigation(
            total_eval_selection or average_eval_selection, "eval_cost"
        )


def trends_main() -> None:
    dashboard_utils.set_page_config(page_title=page_name)
    init_page_state()
    app_name = dashboard_utils.render_sidebar()
    if app_name:
        render_trends(app_name)


if __name__ == "__main__":
    trends_main()
