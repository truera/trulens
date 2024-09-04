import json
from typing import Dict, List

import pandas as pd
import plotly.express as px
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

set_page_config(page_title="Compare")
render_sidebar()
app_name = st.session_state[ST_APP_NAME]

MAX_COMPARATORS = 5
MIN_COMPARATORS = 2

if st.session_state.get("compare.n_comparators", None) is None:
    st.session_state["compare.n_comparators"] = 2


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

    df_tab.dataframe(df.set_index("App Version").T)

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

    chart_tab.plotly_chart(
        fig,
    )


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
    for feedback_col_name in feedback_cols:
        diff_col_name = feedback_col_name + " Variance"
        query_col[diff_col_name] = query_col.iloc[
            :, query_col.columns.str.startswith(feedback_col_name)
        ].var(axis=1)

        diff_cols.append(diff_col_name)
    query_col["feedback_variance"] = query_col[diff_cols].sum(axis=1)
    query_col.sort_values("feedback_variance", ascending=False, inplace=True)
    query_col = query_col.set_index("input")
    query_col["Mean Variance"] = query_col[diff_cols].mean(axis=1)

    def highlight_variance(row: pd.Series):
        colors = []
        for col_name, value in row.items():
            if (
                not pd.isna(value)
                and isinstance(col_name, str)
                and "Variance" in col_name
            ):
                transparency = hex(int(value * 255))[2:]
                if len(transparency) == 1:
                    transparency = "0" + transparency
                colors.append(f"background-color: #ffaaaa{transparency}")
            else:
                colors.append("")
        return colors

    query_col = query_col.sort_values(by="Mean Variance", ascending=False)
    with st.expander("Shared Records"):
        st.dataframe(
            query_col[["Mean Variance"] + diff_cols]
            .style.apply(highlight_variance, axis=1)
            .format("{:.2f}")
        )

    selection = st.selectbox(
        "Select record",
        query_col.index,
    )
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
    app_selector_cols: List[DeltaGenerator],
):
    col_data = {}
    app_ids = []
    n_comparators = st.session_state["compare.n_comparators"]
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

        version = app_selector_cols[i].selectbox(
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

    def _increment_comparators():
        if n_comparators < MAX_COMPARATORS:
            st.session_state["compare.n_comparators"] = n_comparators + 1

    def _decrement_comparators():
        if n_comparators > MIN_COMPARATORS:
            st.session_state["compare.n_comparators"] = n_comparators - 1

    app_selector_cols[-1].button(
        "➕",
        disabled=n_comparators >= MAX_COMPARATORS,
        key="add_comparator",
        on_click=_increment_comparators,
    )

    app_selector_cols[-1].button(
        "➖",
        disabled=n_comparators <= MIN_COMPARATORS,
        key="remove_comparator",
        on_click=_decrement_comparators,
    )

    return col_data, app_ids


def render_app_comparison():
    st.title("Compare")
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, _ = render_app_version_filters(app_name)
    st.divider()

    if versions_df.empty:
        st.warning("No versions available for this app.")
        return

    n_comparators = st.session_state["compare.n_comparators"]

    # Layout
    app_selector_cols = st.columns(
        [4] * n_comparators + [1], gap="large", vertical_alignment="top"
    )
    app_feedback_container = st.container()
    st.divider()
    record_selector_container = st.container()
    record_feedback_graph_container = st.container()
    record_feedback_selector_container = st.container()
    trace_viewer_container = st.container()

    versions = versions_df["app_version"].tolist()

    col_data, app_ids = _version_selectors(
        versions_df, versions, app_selector_cols
    )
    feedback_col_names = _feedback_cols_intersect(col_data)

    _, feedback_directions = get_feedback_defs()

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
    with record_feedback_graph_container:
        _render_all_app_feedback_plot(record_data, feedback_col_names)

    with record_feedback_selector_container:
        if selected_ff := _render_feedback_pills(
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
            key_prefix="shared",
        ):
            feedback_selector_cols = record_feedback_selector_container.columns(
                n_comparators
            )
            for i, app_id in enumerate(app_ids):
                with feedback_selector_cols[i]:
                    record_df = record_data[app_id]["records"]
                    selected_row = record_df.iloc[0]
                    _render_feedback_call(
                        selected_ff, selected_row, feedback_directions
                    )

    with trace_viewer_container:
        trace_cols = trace_viewer_container.columns(n_comparators)
        for i, app_id in enumerate(app_ids):
            with trace_cols[i]:
                record_df = record_data[app_id]["records"]
                selected_row = record_df.iloc[0]
                record_viewer(
                    json.loads(selected_row["record_json"]),
                    json.loads(selected_row["app_json"]),
                    key=f"compare_{app_id}",
                )


if __name__ == "__main__":
    render_app_comparison()
