from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.components.record_viewer_otel import record_viewer_otel
from trulens.dashboard.constants import EXTERNAL_APP_COL_NAME
from trulens.dashboard.constants import HIDE_RECORD_COL_NAME
from trulens.dashboard.constants import PINNED_COL_NAME
from trulens.dashboard.constants import RECORDS_PAGE_NAME as page_name
from trulens.dashboard.utils import streamlit_compat
from trulens.dashboard.utils.dashboard_utils import ST_RECORDS_LIMIT
from trulens.dashboard.utils.dashboard_utils import _get_event_otel_spans
from trulens.dashboard.utils.dashboard_utils import _show_no_records_error
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import is_sis_compatibility_enabled
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills
from trulens.dashboard.utils.streamlit_compat import st_code
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux.styles import aggrid_css
from trulens.dashboard.ux.styles import cell_rules
from trulens.dashboard.ux.styles import default_direction
from trulens.dashboard.ux.styles import radio_button_css


def init_page_state():
    if st.session_state.get(f"{page_name}.initialized", False):
        return

    read_query_params_into_session_state(
        page_name=page_name,
        transforms={
            "app_ids": lambda x: x.split(","),
        },
    )

    app_ids: Optional[List[str]] = st.session_state.get(
        f"{page_name}.app_ids", None
    )
    if app_ids and not st.query_params.get("app_ids", None):
        st.query_params["app_ids"] = ",".join(app_ids)

    st.session_state[f"{page_name}.initialized"] = True


def _format_cost(cost: float, currency: str) -> str:
    if currency == "USD":
        return f"${cost:.2f}"
    else:
        return f"{cost:.3g} {currency}"


def _render_record_metrics(
    records_df: pd.DataFrame, selected_row: pd.Series
) -> None:
    """Render record level metrics (e.g. total tokens, cost, latency) compared
    to the average when appropriate."""

    app_specific_df = records_df[records_df["app_id"] == selected_row["app_id"]]

    token_col, cost_col, latency_col, _ = st_columns([1, 1, 1, 3])

    num_tokens = selected_row["total_tokens"]
    with token_col.container(height=128, border=True):
        st.metric(
            label="Total tokens (#)",
            value=num_tokens,
            help="Number of tokens generated for this record.",
        )

    cost = selected_row["total_cost"]
    cost_currency = selected_row["cost_currency"]
    average_cost = app_specific_df[
        app_specific_df["cost_currency"] == cost_currency
    ]["total_cost"].mean()
    delta_cost = cost - average_cost

    with cost_col.container(height=128, border=True):
        st.metric(
            label=f"Total cost ({cost_currency})",
            value=_format_cost(cost, cost_currency),
            delta=f"{delta_cost:.3g} {cost_currency}"
            if delta_cost != 0
            else None,
            delta_color="inverse",
            help=f"Cost of the app execution measured in {cost_currency}. Delta is relative to average cost for the app."
            if delta_cost != 0
            else f"Cost of the app execution measured in {cost_currency}.",
        )

    latency = selected_row["latency"]
    average_latency = app_specific_df["latency"].mean()
    delta_latency = latency - average_latency
    with latency_col.container(height=128, border=True):
        st.metric(
            label="Latency (s)",
            value=f"{selected_row['latency']}s",
            delta=f"{delta_latency:.3g}s" if delta_latency != 0 else None,
            delta_color="inverse",
            help="Latency of the app execution. Delta is relative to average latency for the app."
            if delta_latency != 0
            else "Latency of the app execution.",
        )


@streamlit_compat.st_fragment
def _render_trace(
    selected_row: pd.Series,
    records_df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    # Start the record specific section
    st.divider()

    # Breadcrumbs
    st.caption(
        f"{selected_row['app_name']} / {selected_row['app_version']} / Record: {selected_row['record_id']}"
    )
    st.markdown(f"#### Record ID: {selected_row['record_id']}")

    input_col, output_col = st_columns(2)
    with input_col.expander("Record Input"):
        st_code(selected_row["input"], wrap_lines=True)

    with output_col.expander("Record Output"):
        st_code(selected_row["output"], wrap_lines=True)

    _render_record_metrics(records_df, selected_row)

    app_json = selected_row["app_json"]
    record_json = selected_row["record_json"]

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

    if is_sis_compatibility_enabled():
        st.subheader("Trace Details")
        st.json(record_json, expanded=1)

        st.subheader("App Details")
        st.json(app_json, expanded=1)
    elif is_otel_tracing_enabled():
        with trace_details:
            st.subheader("Trace Details")
            event_spans = _get_event_otel_spans(selected_row["record_id"])
            if event_spans:
                record_viewer_otel(spans=event_spans, key=None)
            else:
                st.warning("No trace data available for this record.")
    else:
        with trace_details:
            st.subheader("Trace Details")
            record_viewer(record_json, app_json)


def _preprocess_df(
    records_df: pd.DataFrame,
    record_query: Optional[str] = None,
):
    if HIDE_RECORD_COL_NAME in records_df.columns:
        records_df = records_df[~records_df[HIDE_RECORD_COL_NAME]]
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
    records_df["ts"] = pd.to_datetime(records_df["ts"])
    return records_df


def _build_grid_options(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    from st_aggrid.grid_options_builder import GridOptionsBuilder

    gb = GridOptionsBuilder.from_dataframe(df, headerHeight=50)

    gb.configure_column(
        "app_version",
        header_name="App Version",
        resizable=True,
        pinned="left",
        flex=3,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "app_name",
        header_name="App Name",
        hide=True,
        filter="agMultiColumnFilter",
    )

    gb.configure_column(
        PINNED_COL_NAME,
        header_name="Pinned",
        hide=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        EXTERNAL_APP_COL_NAME,
        header_name="External App",
        hide=True,
        filter="agSetColumnFilter",
    )

    gb.configure_column(
        "app_id", header_name="App ID", hide=True, filter="agSetColumnFilter"
    )
    gb.configure_column(
        "record_id",
        header_name="Record ID",
        pinned="left",
        hide=True,
        filter="agSetColumnFilter",
    )

    gb.configure_column(
        "input",
        header_name="Record Input",
        wrapText=True,
        autoHeight=True,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "output",
        header_name="Record Output",
        wrapText=True,
        autoHeight=True,
        filter="agMultiColumnFilter",
    )

    gb.configure_column(
        "record_metadata",
        header_name="Record Metadata",
        filter="agTextColumnFilter",
        valueGetter="JSON.stringify(data.record_metadata)",
    )

    gb.configure_column(
        "total_tokens",
        header_name="Total Tokens (#)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "total_cost",
        header_name="Total Cost (App)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "cost_currency",
        header_name="Cost Currency",
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "latency",
        header_name="Latency (Seconds)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "tags", header_name="Application Tag", filter="agSetColumnFilter"
    )
    gb.configure_column(
        "ts", header_name="Time Stamp", sort="desc", filter="agDateColumnFilter"
    )

    gb.configure_column(
        "feedback_id",
        header_name="Feedback ID",
        hide=True,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "type", header_name="App Type", hide=True, filter="agMultiColumnFilter"
    )
    gb.configure_column(
        "record_json",
        header_name="Record JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "app_json",
        header_name="App JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "cost_json",
        header_name="Cost JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "perf_json",
        header_name="Perf. JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "num_events",
        header_name="Number of Events",
    )

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
            gb.configure_column(
                feedback_col + " direction",
                cellDataType="text",
                valueGetter=f"data['{feedback_col} direction'] ? 'Higher is Better' : 'Lower is Better'",
                hide=True,
            )

    for col in df.columns:
        if "feedback cost" in col:
            gb.configure_column(
                col,
                hide=True,
            )
    gb.configure_grid_options(
        rowHeight=40,
        suppressContextMenu=True,
    )
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()

    return gb.build()


def _render_grid(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    if not is_sis_compatibility_enabled():
        try:
            import st_aggrid
            from st_aggrid.shared import ColumnsAutoSizeMode
            from st_aggrid.shared import DataReturnMode

            event = st_aggrid.AgGrid(
                df,
                gridOptions=_build_grid_options(
                    df=df,
                    feedback_col_names=feedback_col_names,
                    feedback_directions=feedback_directions,
                    version_metadata_col_names=version_metadata_col_names,
                ),
                update_on=["selectionChanged"],
                custom_css={**aggrid_css, **radio_button_css},
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                data_return_mode=DataReturnMode.FILTERED,
                allow_unsafe_jscode=True,
            )
            return pd.DataFrame(event.selected_rows)

        except ImportError:
            # Fallback to st.dataframe if st_aggrid is not installed
            pass

    column_order = [
        "app_version",
        "input",
        "output",
        "record_metadata",
        "total_tokens",
        "total_cost",
        "cost_currency",
        "latency",
        "tags",
        "ts",
        *version_metadata_col_names,
        *feedback_col_names,
        "record_json",
    ]
    column_order = [col for col in column_order if col in df.columns]
    event = st.dataframe(
        df[column_order],
        column_order=column_order,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
        use_container_width=True,
    )
    return df.iloc[event.selection["rows"]]


def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
):
    selected_records = _render_grid(
        df,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
        version_metadata_col_names=version_metadata_col_names,
    )

    if selected_records.empty:
        selected_record_id = st.session_state.get(
            f"{page_name}.selected_record", None
        )
        selected_records = df[df["record_id"] == selected_record_id]

    if selected_records.empty:
        st.info(
            "Click a record's checkbox to view details.",
            icon="ℹ️",
        )
        return

    selected_record = selected_records.iloc[0]
    st.session_state[f"{page_name}.selected_record"] = selected_record[
        "record_id"
    ]
    st.query_params["selected_record"] = selected_record["record_id"]
    _render_trace(selected_record, df, feedback_col_names, feedback_directions)


def _reset_app_ids():
    if f"{page_name}.app_ids" in st.session_state:
        del st.session_state[f"{page_name}.app_ids"]
    if "app_ids" in st.query_params:
        st.query_params.pop("app_ids")


def _render_app_id_args_filter(versions_df: pd.DataFrame):
    if app_ids := st.session_state.get(f"{page_name}.app_ids", None):
        # Reading session state from other pages
        valid_app_ids = versions_df["app_id"].unique()
        if not set(app_ids).issubset(valid_app_ids):
            ids_str = "**`" + "`**, **`".join(app_ids) + "`**"
            st.error(f"Got invalid App IDs: {ids_str}")
            _reset_app_ids()
            return versions_df

        ids_str = "**`" + "`**, **`".join(app_ids) + "`**"
        info_col, show_all_col = st_columns(
            [0.9, 0.1], vertical_alignment="center"
        )
        info_col.info(f"Filtering with App IDs: {ids_str}")
        versions_df = versions_df[versions_df["app_id"].isin(app_ids)]

        show_all_col.button(
            "Show All", use_container_width=True, on_click=_reset_app_ids
        )
        if not st.query_params.get("app_ids", None):
            st.query_params["app_ids"] = ",".join(app_ids)
    return versions_df


def _handle_record_query_change():
    value = st.session_state.get(f"{page_name}.record_search", None)
    if value:
        st.query_params["record_search"] = value
    elif "record_search" in st.query_params:
        st.query_params.pop("record_search")


def render_records(app_name: str):
    """Renders the records page.

    Args:
        app_name (str): The name of the app to render records for.
    """
    st.title(page_name)
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    st.session_state.setdefault(f"{page_name}.record_search", "")
    record_query = st.text_input(
        "Search Records",
        key=f"{page_name}.record_search",
        on_change=_handle_record_query_change,
    )
    versions_df, version_metadata_col_names = render_app_version_filters(
        app_name,
        {f"{page_name}.record_search": record_query},
        page_name_keys=[f"{page_name}.record_search"],
    )

    st.divider()

    versions_df = _render_app_id_args_filter(versions_df)

    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_limit = st.session_state.get(ST_RECORDS_LIMIT, None)
    records_df, feedback_col_names = get_records_and_feedback(
        app_ids=app_ids, app_name=app_name, limit=records_limit
    )

    feedback_col_names = list(feedback_col_names)

    # Preprocess data
    df = _preprocess_df(
        records_df,
        record_query=record_query,
    )

    if df.empty:
        if record_query:
            st.error(f"No records found for search query `{record_query}`.")
        elif app_ids := st.session_state.get(f"{page_name}.app_ids", None):
            app_versions = versions_df[versions_df["app_id"].isin(app_ids)][
                "app_version"
            ].unique()
            versions_str = "**`" + "`**, **`".join(app_versions) + "`**"
            st.error(f"No records found for app version(s): {versions_str}.")
        else:
            # Check for cross-format records before showing generic error
            _show_no_records_error(app_name=app_name, app_ids=app_ids)
        return
    elif records_limit is not None and len(records_df) >= records_limit:
        cols = st_columns([0.9, 0.1], vertical_alignment="center")
        cols[0].info(
            f"Limiting to the latest {records_limit} records. Use the search bar and filters to narrow your search.",
            icon="ℹ️",
        )

        def handle_show_all():
            st.session_state[ST_RECORDS_LIMIT] = None
            if ST_RECORDS_LIMIT in st.query_params:
                del st.query_params[ST_RECORDS_LIMIT]

        cols[1].button(
            "Show all",
            use_container_width=True,
            on_click=handle_show_all,
            help="Show all records. This may take a while.",
        )

    else:
        st.success(f"Found {len(df)} records.")

    _, feedback_directions = get_feedback_defs()

    _render_grid_tab(
        df,
        feedback_col_names,
        feedback_directions,
        version_metadata_col_names,
    )


def records_main():
    set_page_config(page_title=page_name)
    init_page_state()
    app_name = render_sidebar()
    if app_name:
        render_records(app_name)


if __name__ == "__main__":
    records_main()
