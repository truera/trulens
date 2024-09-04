import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.utils.dashboard_utils import ST_APP_NAME
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills

set_page_config(page_title="Records")
render_sidebar()
app_name = st.session_state[ST_APP_NAME]

n_comparators = 2


def _preprocess_df(records_df: pd.DataFrame) -> pd.DataFrame:
    records_df["input"] = (
        records_df["input"].str.encode("utf-8").str.decode("unicode-escape")
    ).astype(str)
    records_df["output"] = (
        records_df["output"].str.encode("utf-8").str.decode("unicode-escape")
    ).astype(str)
    return records_df


def _render_feedback_results(
    selected_input: str,
    col_data: dict[str, Any],
    versions_df: pd.DataFrame,
    feedback_directions: Dict[str, bool],
    app_id: str,
):
    record_df = col_data["records"]
    feedback_col_names = col_data["feedback_cols"]
    selected_row = record_df[record_df["input"] == selected_input].iloc[0]

    # Feedback results
    st.subheader("Feedback results")

    if feedback_col := _render_feedback_pills(
        feedback_col_names, selected_row, feedback_directions
    ):
        _render_feedback_call(feedback_col, selected_row, feedback_directions)


def _render_trace_details(
    col_data: dict[str, Any], selected_input: str, app_id: str
):
    record_df = col_data["records"]
    selected_row = record_df[record_df["input"] == selected_input].iloc[0]
    app_json = selected_row["app_json"]

    # Trace details
    st.subheader("Trace details")
    record_viewer(
        json.loads(selected_row["record_json"]),
        json.loads(app_json),
        key=f"compare_{app_id}",
    )


def _feedback_cols_intersect(
    col_data: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    feedback_cols = None
    for _, data in col_data.items():
        if feedback_cols is None:
            feedback_cols = set(data["feedback_cols"])
        else:
            feedback_cols = feedback_cols.intersection(data["feedback_cols"])
    if feedback_cols is None:
        return []
    return list(feedback_cols)


def _compute_feedback_means(
    feedback_cols: List[str], col_data: Dict[str, Dict[str, pd.DataFrame]]
):
    feedback_means = {}

    for feedback_col_name in feedback_cols:
        col_values = {
            app_id: data["records"][feedback_col_name]
            for app_id, data in col_data.items()
        }
        total_mean = pd.concat(list(col_values.values())).mean()
        feedback_means[feedback_col_name] = round(total_mean, 3)
    return feedback_means
    #     for app_id, col in col_values.items():
    #         app_mean = col.mean()
    #         delta = app_mean - total_mean
    #         mean_feedback_values[app_id] = app_mean
    # return mean_feedback_values


def _render_app_feedback_means(
    app_id: str,
    feedback_means: Dict[str, float],
    feedback_cols: List[str],
    col_data: Dict[str, Dict[str, pd.DataFrame]],
    feedback_directions: Dict[str, bool],
    default_n_cols: int = 4,
):
    n_cols = min(default_n_cols, len(feedback_cols))
    cols = st.columns(n_cols)
    for i, feedback_col_name in enumerate(feedback_cols):
        feedback_data = col_data[app_id]["records"][feedback_col_name]
        app_mean = round(feedback_data.mean(), 3)
        total_mean = feedback_means[feedback_col_name]
        delta = round(app_mean - total_mean, 3)

        with cols[i % n_cols]:
            with st.container(border=True):
                st.metric(
                    label=feedback_col_name,
                    value=app_mean,
                    delta=delta,
                    delta_color="normal"
                    if feedback_directions.get(feedback_col_name, True)
                    else "inverse",
                    help=f"Total mean: {total_mean}",
                )


@st.fragment
def _render_shared(
    col_data: Dict[str, Dict[str, pd.DataFrame]],
    feedback_cols: List[str],
    versions_df: pd.DataFrame,
):
    query_col = None
    cols = ["input"] + feedback_cols
    for app_id, data in col_data.items():
        if query_col is None:
            query_col = data["records"][cols].rename(
                columns=lambda x: f"{x}_{app_id}" if x in feedback_cols else x
            )
        else:
            query_col = query_col.merge(
                data["records"][cols],
                how="inner",
                on="input",
            ).rename(
                columns=lambda x: f"{x}_{app_id}" if x in feedback_cols else x
            )
    assert query_col is not None

    if query_col.empty:
        st.write("No shared records found.")
        return
    # Feedback difference
    diff_cols = []
    for feedback_col_name in feedback_cols:
        diff_col_name = feedback_col_name + " Variance"
        query_col[diff_col_name] = query_col.iloc[
            :, query_col.columns.str.startswith(feedback_col_name)
        ].var(axis=1)

        diff_cols.append(diff_col_name)
    query_col["feedback_variance"] = query_col[diff_cols].sum(axis=1)
    query_col.sort_values("feedback_variance", ascending=False, inplace=True)
    query_col = query_col.set_index("input")

    st.write(query_col[diff_cols])
    selection = st.selectbox("Select record", query_col.index)
    return selection


def _handle_app_version_search_params(i: int):
    def update():
        app_id = st.session_state[f"compare_app_version_{i}"]
        if not st.query_params.get(f"app_version_{i}", None):
            st.query_params[f"app_version_{i}"] = app_id

    return update


def _version_selectors(
    versions_df: pd.DataFrame,
    versions: list[str],
    dropdown_cols: List[DeltaGenerator],
):
    col_data = {}
    app_ids = []
    for i in range(n_comparators):
        app_id = st.query_params.get(f"app_version_{i}", None)
        if app_id and app_id in versions_df["app_id"].values:
            version = versions_df[versions_df["app_id"] == app_id][
                "app_version"
            ].values[0]
            idx = versions.index(version)
        else:
            idx = i
            app_id = versions_df.iloc[idx]["app_id"]
            version = versions_df.iloc[idx]["app_version"]

        version = dropdown_cols[i].selectbox(
            f"App Version {i}",
            key=f"compare_app_version_{i}",
            options=versions,
            index=idx,
        )
        if version:
            app_id = versions_df[versions_df["app_version"] == version][
                "app_id"
            ].values[0]
            app_ids.append(app_id)
            _handle_app_version_search_params(i)()
            records, feedback_cols = get_records_and_feedback(app_ids=[app_id])
            col_data[app_id] = {
                "version": version,
                "records": _preprocess_df(records),
                "feedback_cols": feedback_cols,
            }
    return col_data, app_ids


def render_app_comparison():
    st.title("Records")
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, _ = render_app_version_filters(app_name)
    st.divider()

    if versions_df.empty:
        st.write("No versions available for this app.")
        return

    dropdown_cols = st.columns(n_comparators, gap="large")
    shared_cont = st.container()
    feedback_compare_cols = st.columns(n_comparators, gap="large")
    trace_compare_cols = st.columns(n_comparators, gap="large")
    versions = versions_df["app_version"].tolist()

    col_data, app_ids = _version_selectors(versions_df, versions, dropdown_cols)
    feedback_cols = _feedback_cols_intersect(col_data)

    _, feedback_directions = get_feedback_defs()
    feedback_means = _compute_feedback_means(feedback_cols, col_data)

    for i, app_id in enumerate(app_ids):
        with dropdown_cols[i]:
            _render_app_feedback_means(
                app_id=app_id,
                feedback_means=feedback_means,
                feedback_cols=feedback_cols,
                feedback_directions=feedback_directions,
                col_data=col_data,
            )

    with shared_cont:
        if selected_input := _render_shared(
            col_data, feedback_cols, versions_df
        ):
            for i, app_id in enumerate(app_ids):
                with feedback_compare_cols[i]:
                    _render_feedback_results(
                        selected_input,
                        col_data=col_data[app_id],
                        versions_df=versions_df,
                        feedback_directions=feedback_directions,
                        app_id=app_id,
                    )

            st.divider()

            for i in range(n_comparators):
                with trace_compare_cols[i]:
                    _render_trace_details(
                        col_data=col_data[app_ids[i]],
                        selected_input=selected_input,
                        app_id=app_ids[i],
                    )


if __name__ == "__main__":
    render_app_comparison()
