import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.constants import COMPARE_PAGE_NAME as page_name
from trulens.dashboard.utils.dashboard_utils import ST_APP_NAME
from trulens.dashboard.utils.dashboard_utils import add_query_param
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills

MAX_COMPARATORS = 5
MIN_COMPARATORS = 2
DEFAULT_COMPARATORS = MIN_COMPARATORS


def init_page_state(app_name: str):
    if st.session_state.get(f"{page_name}.initialized", False):
        return

    if app_name:
        add_query_param(ST_APP_NAME, app_name)

    # Read app_ids from query args if passed in
    read_query_params_into_session_state(
        page_name=page_name,
        transforms={
            "app_ids": lambda x: x.split(","),
        },
    )

    # If app_ids is passed through session state
    app_ids: Optional[List[str]] = st.session_state.get(
        f"{page_name}.app_ids", None
    )
    if not app_ids:
        # to be inferred later
        st.session_state[f"{page_name}.app_ids"] = [None, None]
    elif not st.query_params.get("app_ids", None):
        st.query_params["app_ids"] = ",".join(app_ids)

    st.session_state[f"{page_name}.initialized"] = True


def _preprocess_df(records_df: pd.DataFrame) -> pd.DataFrame:
    records_df["input"] = (
        records_df["input"].str.encode("utf-8").str.decode("unicode-escape")
    ).astype(str)
    records_df["output"] = (
        records_df["output"].str.encode("utf-8").str.decode("unicode-escape")
    ).astype(str)
    return records_df


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


def _render_all_app_feedback_plot(
    col_data: Dict[str, Dict[str, pd.DataFrame]], feedback_cols: List[str]
):
    ff_dfs = []
    for _, data in col_data.items():
        app_df = data["records"]
        df = app_df[feedback_cols].mean(axis=0)
        df["App Version"] = data["version"]
        ff_dfs.append(df)

    df = pd.concat(ff_dfs, axis=1).T
    chart_tab, df_tab = st.tabs(["Graph", "DataFrame"])

    _df = df.set_index("App Version").T
    if len(_df.columns) == 2:
        diff_col = "Diff"
        _df[diff_col] = np.abs(_df[_df.columns[0]] - _df[_df.columns[1]])
    else:
        diff_col = "Variance"
        _df[diff_col] = _df.var(axis=1)

    df_tab.dataframe(
        _df.sort_values(diff_col, ascending=False)
        .style.apply(_highlight_variance, axis=1)
        .format("{:.3f}"),
        use_container_width=True,
    )

    df = df.melt(
        id_vars="App Version",
        var_name="Feedback Function Name",
        value_name="Feedback Function Values",
    )
    fig = px.histogram(
        data_frame=df,
        x="Feedback Function Name",
        y="Feedback Function Values",
        color="App Version",
        barmode="group",
        histfunc="avg",
    )
    fig.update_layout(dragmode=False)
    fig.update_yaxes(fixedrange=True, range=[0, 1])
    fig.update_xaxes(fixedrange=True)

    chart_tab.plotly_chart(fig, use_container_width=True)


def _highlight_variance(row: pd.Series):
    colors = []
    for col_name, value in row.items():
        if (
            not pd.isna(value)
            and isinstance(col_name, str)
            and (col_name.endswith("Variance") or col_name.endswith("Diff"))
        ):
            transparency = hex(int(value * 255))[2:]
            if len(transparency) == 1:
                transparency = "0" + transparency
            colors.append(f"background-color: #ffaaaa{transparency}")
        else:
            colors.append("")
    return colors


def _render_shared_records(
    col_data: Dict[str, Dict[str, pd.DataFrame]],
    feedback_cols: List[str],
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
        st.warning("No shared records found.")
        return
    # Feedback difference
    diff_cols = []
    if len(col_data) == 2:
        col_suffix = "Diff"
    else:
        col_suffix = "Variance"
    for feedback_col_name in feedback_cols:
        diff_col_name = f"{feedback_col_name} {col_suffix}"
        diff_cols.append(diff_col_name)

        if len(col_data) == 2:
            query_col[diff_col_name] = np.abs(
                query_col.iloc[
                    :, query_col.columns.str.startswith(feedback_col_name)
                ]
                .diff(axis=1)
                .iloc[:, 1]
            )
        else:
            diff_col_name = feedback_col_name + " Variance"
            query_col[diff_col_name] = query_col.iloc[
                :, query_col.columns.str.startswith(feedback_col_name)
            ].var(axis=1)

    query_col = query_col.set_index("input")
    agg_diff_col = f"Mean {col_suffix}"
    query_col[agg_diff_col] = query_col[diff_cols].mean(axis=1)

    query_col = query_col.sort_values(by=agg_diff_col, ascending=False)
    with st.expander("Shared Records Stats"):
        st.dataframe(
            query_col[[agg_diff_col] + diff_cols]
            .style.apply(_highlight_variance, axis=1)
            .format("{:.3f}"),
            use_container_width=True,
        )

    selection = st.selectbox(
        "Select record",
        query_col.index,
    )
    return selection


def _lookup_app_version(
    versions_df: pd.DataFrame,
    app_version: Optional[str] = None,
    app_id: Optional[str] = None,
):
    if app_version and app_id:
        raise ValueError("Can only pass one of `app_id` or `app_version`")
    elif app_version:
        return versions_df[versions_df["app_version"] == app_version].iloc[0]
    elif app_id:
        return versions_df[versions_df["app_id"] == app_id].iloc[0]
    else:
        raise ValueError("Must pass one of `app_id` or `app_version`")


def _render_version_selectors(
    versions_df: pd.DataFrame,
):
    def _increment_comparators(selected_app_ids: List[Optional[str]]):
        if len(selected_app_ids) < MAX_COMPARATORS:
            selected_app_ids.append(None)
            st.session_state[f"{page_name}.app_ids"] = selected_app_ids

    def _decrement_comparators(selected_app_ids: List[Optional[str]]):
        if len(selected_app_ids) > MIN_COMPARATORS:
            selected_app_ids.pop(-1)
            st.session_state[f"{page_name}.app_ids"] = selected_app_ids

    current_app_ids = [
        app_id
        for app_id in st.session_state.get(f"{page_name}.app_ids", [])[
            :MAX_COMPARATORS
        ]
    ]

    versions = versions_df["app_version"].tolist()
    # pinned_versions = None
    # if PINNED_COL_NAME in versions_df.columns:
    #     pinned_versions = set(
    #         versions_df[versions_df[PINNED_COL_NAME]]["app_version"].unique()
    #     )

    inc_col, dec_col, _ = st.columns([0.15, 0.15, 0.7])
    inc_col.button(
        "➕ Add App Version",
        disabled=len(current_app_ids) >= MAX_COMPARATORS,
        key="add_comparator",
        use_container_width=True,
        on_click=_increment_comparators,
        args=(current_app_ids,),
    )

    dec_col.button(
        "➖ Remove App Version",
        disabled=len(current_app_ids) <= MIN_COMPARATORS,
        key="remove_comparator",
        use_container_width=True,
        on_click=_decrement_comparators,
        args=(current_app_ids,),
    )

    with st.form("app_version_selector_form", border=False):
        app_selector_cols = st.columns(
            len(current_app_ids),
            gap="large",
            vertical_alignment="top",
        )

        for i, app_id in enumerate(current_app_ids):
            if app_id and app_id in versions_df["app_id"].values:
                version = _lookup_app_version(versions_df, app_id=app_id)[
                    "app_version"
                ]
                idx = versions.index(version)
            else:
                idx = i
                app_id = versions_df.iloc[idx]["app_id"]
                version = versions_df.iloc[idx]["app_version"]
                current_app_ids[i] = app_id

            select_container = app_selector_cols[i].container()
            # filter_container = app_selector_cols[i].container()

            select_options = versions
            # if pinned_versions:
            #     checkbox_key = (
            #         f"{page_name}.app_version_selector_{i}.show_pinned"
            #     )
            #     if filter_container.checkbox(
            #         "Only Show Pinned", key=checkbox_key
            #     ):
            #         select_options = list(pinned_versions)

            selectbox_key = f"{page_name}.app_version_selector_{i}"
            if version := select_container.selectbox(
                f"App Version {i}",
                key=selectbox_key,
                options=select_options,
                index=idx,
            ):
                app_id = _lookup_app_version(versions_df, app_version=version)[
                    "app_id"
                ]
                current_app_ids[i] = app_id

        if st.form_submit_button("Apply"):
            st.session_state[f"{page_name}.app_ids"] = current_app_ids
            st.query_params["app_ids"] = ",".join(
                str(app_id) for app_id in current_app_ids
            )
            records, feedback_cols = get_records_and_feedback(
                app_ids=current_app_ids
            )
            records = _preprocess_df(records)
            col_data = {
                app_id: {
                    "version": app_df["app_version"].unique()[0],
                    "records": app_df,
                    "feedback_cols": [
                        col for col in feedback_cols if col in app_df.columns
                    ],
                }
                for app_id, app_df in records.groupby(by="app_id")
            }

            st.session_state[f"{page_name}.col_data"] = col_data


def render_app_comparison(app_name: str):
    st.title(page_name)
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, _ = render_app_version_filters(app_name)

    global MAX_COMPARATORS
    MAX_COMPARATORS = min(MAX_COMPARATORS, len(versions_df))

    if versions_df.empty:
        st.error(f"No app versions available for app `{app_name}`.")
        return
    elif len(versions_df) < MIN_COMPARATORS:
        st.error(
            "Not enough App Versions found to compare. Try a different page instead."
        )

    # get app version and record data
    _render_version_selectors(versions_df)
    col_data = st.session_state.get(f"{page_name}.col_data", None)
    if not col_data:
        return

    st.divider()
    feedback_col_names = _feedback_cols_intersect(col_data)
    _, feedback_directions = get_feedback_defs()

    # Layout
    app_feedback_container = st.container()
    st.divider()
    record_selector_container = st.container()
    record_header_container = st.container()
    record_feedback_graph_container = st.container()
    record_feedback_selector_container = st.container()
    trace_viewer_container = st.container()

    with app_feedback_container:
        app_feedback_container.header("Shared Feedback Functions")
        _render_all_app_feedback_plot(col_data, feedback_col_names)

    with record_selector_container:
        record_selector_container.header("Shared Records")
        selected_input = _render_shared_records(col_data, feedback_col_names)

    if selected_input is None:
        return

    record_data = {
        app_id: {
            **data,
            "records": data["records"][
                data["records"]["input"] == selected_input
            ],
        }
        for app_id, data in col_data.items()
    }
    record_header_container.divider()
    record_header_container.header("Record Comparison")
    with record_feedback_graph_container:
        _render_all_app_feedback_plot(record_data, feedback_col_names)

    with record_feedback_selector_container:
        if selected_ff := _render_feedback_pills(
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
            key_prefix="shared",
        ):
            feedback_selector_cols = record_feedback_selector_container.columns(
                len(record_data), gap="large"
            )
            for i, app_id in enumerate(record_data):
                with feedback_selector_cols[i]:
                    record_df = record_data[app_id]["records"]
                    selected_row = record_df.iloc[0]
                    _render_feedback_call(
                        selected_ff, selected_row, feedback_directions
                    )

    with trace_viewer_container:
        trace_cols = trace_viewer_container.columns(
            len(record_data), gap="large"
        )
        for i, app_id in enumerate(record_data):
            with trace_cols[i]:
                record_df = record_data[app_id]["records"]
                selected_row = record_df.iloc[0]
                record_viewer(
                    json.loads(selected_row["record_json"]),
                    json.loads(selected_row["app_json"]),
                    key=f"compare_{app_id}",
                )


if __name__ == "__main__":
    set_page_config(page_title=page_name)
    app_name = render_sidebar()
    if app_name:
        init_page_state(app_name)
        render_app_comparison(app_name)
