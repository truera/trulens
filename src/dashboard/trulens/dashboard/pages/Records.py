import json
from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import ColumnsAutoSizeMode
from st_aggrid.shared import DataReturnMode
import streamlit as st
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.utils.dashboard_utils import ST_APP_NAME
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills
from trulens.dashboard.ux.styles import cell_rules
from trulens.dashboard.ux.styles import cell_rules_styles
from trulens.dashboard.ux.styles import default_direction

set_page_config(page_title="Records")
render_sidebar()
app_name = st.session_state[ST_APP_NAME]


def _render_record_metrics(
    records_df: pd.DataFrame, selected_row: pd.DataFrame
) -> None:
    """Render record level metrics (e.g. total tokens, cost, latency) compared
    to the average when appropriate."""

    app_specific_df = records_df[records_df["app_id"] == selected_row["app_id"]]

    token_col, cost_col, latency_col = st.columns(3)

    num_tokens = selected_row["total_tokens"]
    token_col.metric(label="Total tokens (#)", value=num_tokens)

    cost = selected_row["total_cost"]
    average_cost = app_specific_df["total_cost"].mean()
    delta_cost = f"{cost - average_cost:.3g}"
    cost_col.metric(
        label="Total cost (USD)",
        value=selected_row["total_cost"],
        delta=delta_cost,
        delta_color="inverse",
    )

    latency = selected_row["latency"]
    average_latency = app_specific_df["latency"].mean()
    delta_latency = f"{latency - average_latency:.3g}s"
    latency_col.metric(
        label="Latency (s)",
        value=selected_row["latency"],
        delta=delta_latency,
        delta_color="inverse",
    )


def _render_trace(
    selected_rows: pd.DataFrame,
    records_df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    # Start the record specific section
    st.divider()

    # Breadcrumbs
    selected_row = selected_rows.iloc[0]
    st.caption(f"{selected_row['app_id']} / {selected_row['record_id']}")
    st.header(f"{selected_row['record_id']}")

    _render_record_metrics(records_df, selected_row)

    app_json = json.loads(selected_row["app_json"])
    record_json = json.loads(selected_row["record_json"])

    feedback_results = st.container()
    trace_details = st.container()

    # Feedback results
    feedback_results.subheader("Feedback Results")
    with feedback_results:
        if selected_ff := _render_feedback_pills(
            feedback_col_names=feedback_col_names,
            selected_row=selected_row,
            feedback_directions=feedback_directions,
        ):
            _render_feedback_call(
                selected_ff,
                selected_row,
                feedback_directions=feedback_directions,
            )

    # Trace details
    with trace_details:
        st.subheader("Trace Details")
        record_viewer(record_json, app_json)


def _preprocess_df(
    records_df: pd.DataFrame,
    record_query: Optional[str] = None,
):
    records_df = records_df.sort_values(by="ts", ascending=False)
    records_df["input"] = (
        records_df["input"].str.encode("utf-8").str.decode("unicode-escape")
    )
    records_df["output"] = (
        records_df["output"].str.encode("utf-8").str.decode("unicode-escape")
    )

    # Reorder App columns to the front. Used for aggrid display
    records_df.insert(0, "app_id", records_df.pop("app_id"))
    records_df.insert(0, "app_name", records_df.pop("app_name"))
    records_df.insert(0, "record_id", records_df.pop("record_id"))
    records_df.insert(0, "app_version", records_df.pop("app_version"))

    # if "app_json" in records_df.columns:
    #     records_df = records_df.drop(columns="app_json")

    if record_query:
        records_df = records_df[
            records_df["app_version"].str.contains(record_query, case=False)
            | records_df["input"].str.contains(record_query, case=False)
            | records_df["output"].str.contains(record_query, case=False)
        ]
    return records_df


def _build_grid_options(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_column(
        "app_version",
        header_name="App Version",
        resizable=True,
        pinned="left",
        flex=3,
    )
    gb.configure_column(
        "app_name",
        header_name="App Name",
        hide=True,
    )
    gb.configure_column("app_id", header_name="App ID", hide=True)
    gb.configure_column(
        "record_id", header_name="Record ID", pinned="left", hide=True
    )

    gb.configure_column(
        "input", header_name="User Input", wrapText=True, autoHeight=True
    )
    gb.configure_column(
        "output", header_name="Response", wrapText=True, autoHeight=True
    )
    gb.configure_column("record_metadata", header_name="Record Metadata")

    gb.configure_column("total_tokens", header_name="Total Tokens (#)")
    gb.configure_column("total_cost", header_name="Total Cost (App)")
    gb.configure_column("cost_currency", header_name="Cost Currency")
    gb.configure_column("latency", header_name="Latency (Seconds)")
    gb.configure_column("tags", header_name="Application Tag")
    gb.configure_column("ts", header_name="Time Stamp", sort="desc")

    gb.configure_column("feedback_id", header_name="Feedback ID", hide=True)
    gb.configure_column("type", header_name="App Type", hide=True)
    gb.configure_column("record_json", header_name="Record JSON", hide=True)
    gb.configure_column("app_json", header_name="App JSON", hide=True)
    gb.configure_column("cost_json", header_name="Cost JSON", hide=True)
    gb.configure_column("perf_json", header_name="Perf. JSON", hide=True)

    for metadata_col in version_metadata_col_names:
        gb.configure_column(
            metadata_col,
            header_name=metadata_col,
            resizable=True,
            pinned=True,
            hide=True,
        )

    for feedback_col in feedback_col_names:
        if "distance" in feedback_col:
            gb.configure_column(feedback_col, hide=False)
            gb.configure_column(feedback_col + "_calls", hide=True)
        else:
            feedback_direction = (
                "HIGHER_IS_BETTER"
                if feedback_directions.get(feedback_col, default_direction)
                else "LOWER_IS_BETTER"
            )
            gb.configure_column(
                feedback_col,
                cellClassRules=cell_rules[feedback_direction],
                hide=False,
            )
            gb.configure_column(
                feedback_col + "_calls",
                hide=True,
            )

    for col in df.columns:
        if "feedback cost" in col:
            gb.configure_column(
                col,
                hide=True,
            )
    gb.configure_grid_options(rowHeight=40)
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()
    return gb.build()


# @st.fragment
def _render_grid(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    return AgGrid(
        df,
        key="records_data",
        # height=1200,
        gridOptions=_build_grid_options(
            df=df,
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
            version_metadata_col_names=version_metadata_col_names,
        ),
        update_on=["selectionChanged"],
        custom_css=cell_rules_styles,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        data_return_mode=DataReturnMode.FILTERED,
        allow_unsafe_jscode=True,
    )


def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
):
    grid_data = _render_grid(
        df,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
        version_metadata_col_names=version_metadata_col_names,
    )
    selected_rows = grid_data.selected_rows
    if selected_rows is None or len(selected_rows) == 0:
        st.info(
            "No record selected. Click a record's checkbox to view details.",
            icon="ℹ️",
        )
        return

    selected_record = pd.DataFrame(selected_rows)
    _render_trace(selected_record, df, feedback_col_names, feedback_directions)


def _render_plot_tab(df: pd.DataFrame, feedback_col_names: List[str]):
    cols = 4
    rows = len(feedback_col_names) // cols + 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=feedback_col_names)

    for i, feedback_col_name in enumerate(feedback_col_names):
        row_num = i // cols + 1
        col_num = i % cols + 1
        _df = df[feedback_col_name].dropna()

        plot = go.Histogram(
            x=_df,
            xbins={
                "size": 0.2,
                "start": 0,
                "end": 1.0,
            },
            histfunc="count",
            texttemplate="%{y}",
        )
        fig.add_trace(
            plot,
            row=row_num,
            col=col_num,
        )
    fig.update_layout(height=250 * rows, width=250 * cols, dragmode=False)
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(fixedrange=True, range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_records():
    st.title("Records")
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    record_query = st.text_input("Search Records", key="record_search")
    versions_df, version_metadata_col_names = render_app_version_filters(
        app_name
    )

    if app_ids := st.session_state.get("records.app_ids", None):
        # Reading session state from other pages
        ids_str = "**`" + "`**, **`".join(app_ids) + "`**"
        st.info(f"Filtering with App IDs: {ids_str}")
        versions_df = versions_df[versions_df["app_id"].isin(app_ids)]
        st.session_state["records.app_ids"] = None

    st.divider()

    if versions_df.empty:
        st.warning("No versions available for this app.")
        return
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_df, feedback_col_names = get_records_and_feedback(
        app_ids, limit=1000
    )
    if records_df.empty:
        st.warning("No records available for this app.")
        return
    elif len(records_df) == 1000:
        st.info(
            "Limiting to the latest 1000 records. Use the search bar and filters to narrow your search.",
            icon="ℹ️",
        )

    feedback_col_names = list(feedback_col_names)

    # Preprocess data
    df = _preprocess_df(
        records_df,
        record_query=record_query,
    )
    _, feedback_directions = get_feedback_defs()

    grid_tab, plot_tab = st.tabs(["Records", "Feedback Distribution"])
    with grid_tab:
        _render_grid_tab(
            df,
            feedback_col_names,
            feedback_directions,
            version_metadata_col_names,
        )
    with plot_tab:
        _render_plot_tab(df, feedback_col_names)


if __name__ == "__main__":
    render_records()
