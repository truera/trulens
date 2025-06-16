from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.components.record_viewer_otel import record_viewer_otel
from trulens.dashboard.constants import COMPARE_PAGE_NAME as page_name
from trulens.dashboard.constants import HIDE_RECORD_COL_NAME
from trulens.dashboard.constants import PINNED_COL_NAME
from trulens.dashboard.utils.dashboard_utils import _get_event_otel_spans
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
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux.styles import aggrid_css
from trulens.dashboard.ux.styles import diff_cell_css
from trulens.dashboard.ux.styles import diff_cell_rules
from trulens.dashboard.ux.styles import radio_button_css

MAX_COMPARATORS = 5
MIN_COMPARATORS = 2
DEFAULT_COMPARATORS = MIN_COMPARATORS


def init_page_state():
    if st.session_state.get(f"{page_name}.initialized", False):
        return

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
    fig.update_layout(dragmode=False, barcornerradius=4, bargroupgap=0.05)
    fig.update_yaxes(fixedrange=True, showgrid=False, range=[0, 1])
    fig.update_xaxes(fixedrange=True, showgrid=False)

    chart_tab.plotly_chart(fig, use_container_width=True)


def _highlight_variance(row: pd.Series):
    colors = []
    for col_name, value in row.items():
        if col_name == "input":
            colors.append("")
        elif (
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


def _render_advanced_filters(
    query_col: pd.DataFrame,
    feedback_cols: List[str],
):
    def handle_add_clause():
        st.session_state[f"{page_name}.record_filter.n_clauses"] = (
            st.session_state.get(f"{page_name}.record_filter.n_clauses", 0) + 1
        )

    def handle_clear_clauses(remove_all: bool = True):
        if remove_all:
            st.session_state[f"{page_name}.record_filter.n_clauses"] = 0
        else:
            n_clauses = st.session_state.get(
                f"{page_name}.record_filter.n_clauses", 0
            )
            n_clauses = max(n_clauses - 1, 0)
            st.session_state[f"{page_name}.record_filter.n_clauses"] = n_clauses

    def render_clause(st_cols: List[DeltaGenerator], idx: int = 0):
        feedback_col = st_cols[0].selectbox(
            "Feedback Function",
            feedback_cols,
            label_visibility="collapsed",
            key=f"{page_name}.record_filter.clause_{idx}.sort_by",
        )
        relevant_app_versions = [
            col[len(feedback_col) + 1 :]
            for col in query_col.columns
            if col.startswith(feedback_col)
        ]

        operator = st_cols[2].selectbox(
            "Operator",
            [">", "<", "==", "!=", ">=", "<="],
            label_visibility="collapsed",
            key=f"{page_name}.record_filter.clause_{idx}.operator",
        )
        if len(relevant_app_versions) == 0:
            st.info("No relevant app versions found.")
            return
        elif len(relevant_app_versions) == 1:
            app1 = relevant_app_versions[0]
            st_cols[1].markdown(f"`{app1}`")
            app2 = None
            value = st_cols[1].number_input(
                "Value",
                key=f"{page_name}.record_filter.clause_{idx}.value",
                label_visibility="collapsed",
            )
        elif len(relevant_app_versions) == 2:
            app1 = relevant_app_versions[0]
            app2 = relevant_app_versions[1]
            value = None
            st_cols[1].markdown(f"`{app1}`")
            st_cols[3].markdown(f"`{app2}`")
        else:
            app1 = st_cols[1].selectbox(
                "App 1",
                relevant_app_versions,
                key=f"{page_name}.record_filter.clause_{idx}.app1",
                label_visibility="collapsed",
            )
            app2 = st_cols[3].selectbox(
                "App 2",
                relevant_app_versions,
                key=f"{page_name}.record_filter.clause_{idx}.app2",
                label_visibility="collapsed",
            )
            value = None

        return app1, app2, value, operator, feedback_col

    with (
        st.expander(
            "Advanced Filters",
        ),
        st.form("advanced_filter_form", border=False),
    ):
        out = None

        filters = []
        c1, c2, c3, c4, _ = st_columns([0.15, 0.15, 0.15, 0.15, 0.4])

        n_clauses = st.session_state.get(
            f"{page_name}.record_filter.n_clauses", 0
        )

        if n_clauses:
            st_cols = st_columns(5, vertical_alignment="center")
            for i, header in enumerate([
                "Feedback Function",
                "App Version 0",
                "Operator",
                "App Version 1",
                "",
            ]):
                st_cols[i].write(header)

        for i in range(n_clauses):
            clause_container = st.container()
            st_cols = st_columns(
                5, vertical_alignment="center", container=clause_container
            )
            clause_fields = render_clause(st_cols, idx=i)
            if clause_fields is None:
                # TODO: test this
                continue

            app1, app2, value, operator, feedback_col = clause_fields
            filters.append({
                "app1": app1,
                "app2": app2,
                "value": value,
                "operator": operator,
                "feedback_col": feedback_col,
            })

        submit_button = c1.form_submit_button(
            "Apply Filter",
            type="primary",
            use_container_width=True,
        )
        c2.form_submit_button(
            "Add Clause",
            use_container_width=True,
            on_click=handle_add_clause,
        )
        c3.form_submit_button(
            "Remove Clause",
            use_container_width=True,
            on_click=handle_clear_clauses,
            kwargs={"remove_all": False},
        )
        c4.form_submit_button(
            "Remove All",
            use_container_width=True,
            on_click=handle_clear_clauses,
            kwargs={"remove_all": True},
        )

        if submit_button:
            out = query_col
            for filter_dict in filters:
                app1 = filter_dict["app1"]
                app2 = filter_dict["app2"]
                value = filter_dict["value"]
                operator = filter_dict["operator"]
                feedback_col = filter_dict["feedback_col"]

                if value is not None:
                    out = out[
                        eval(f"out['{feedback_col}_{app1}'] {operator} {value}")
                    ]
                else:
                    out = out[
                        eval(
                            f"out['{feedback_col}_{app1}'] {operator} out['{feedback_col}_{app2}']"
                        )
                    ]

    if out is not None:
        if len(out) == len(query_col):
            st.success(f"Showing all {len(out)} record(s).")
        elif len(out) == 0:
            st.warning("Got 0 records after applying filter.")
        else:
            st.success(f"Filters applied and got {len(out)} record(s).")
        st.session_state[f"{page_name}.record_filter.result"] = out
    # else:
    # st.session_state[f"{page_name}.record_filter.result"] = query_col


def _build_grid_options(
    df: pd.DataFrame,
    agg_diff_col: str,
    diff_cols: List[str],
    record_id_cols: List[str],
    num_comparators: int,
):
    from st_aggrid.grid_options_builder import GridOptionsBuilder

    gb = GridOptionsBuilder.from_dataframe(df, headerHeight=50, flex=1)

    gb.configure_column(
        "input",
        header_name="Input",
        resizable=True,
        wrapText=True,
        pinned="left",
        flex=3,
        filter="agMultiColumnFilter",
    )
    gb.configure_columns(
        record_id_cols,
        hide=True,
        filter="agSetColumnFilter",
    )

    # Determine tooltip text based on number of comparators
    if num_comparators == 2:
        agg_tooltip = "Mean Diff: Average of absolute differences between the two app versions"
        diff_tooltip = "Diff: Absolute difference between the two app versions (|version1 - version2|)"
    elif num_comparators > 2:
        agg_tooltip = f"Mean Std Dev: Average standard deviation across all {num_comparators} app versions"
        diff_tooltip = f"Std Dev: Standard deviation of values across all {num_comparators} app versions"

    gb.configure_column(
        agg_diff_col,
        cellClassRules=diff_cell_rules,
        filter="agNumberColumnFilter",
        headerTooltip=agg_tooltip,
    )
    gb.configure_columns(
        diff_cols,
        cellClassRules=diff_cell_rules,
        filter="agNumberColumnFilter",
        headerTooltip=diff_tooltip,
    )

    gb.configure_grid_options(
        rowHeight=45,
        suppressContextMenu=True,
    )
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True, paginationPageSize=25)
    gb.configure_side_bar(filters_panel=False, columns_panel=False)
    gb.configure_grid_options(autoSizeStrategy={"type": "fitCellContents"})
    return gb.build()


def _render_grid(
    df: pd.DataFrame,
    agg_diff_col: str,
    diff_cols: List[str],
    record_id_cols: List[str],
    num_comparators: int,
    grid_key: Optional[str] = None,
):
    if not is_sis_compatibility_enabled():
        try:
            import st_aggrid

            columns_state = st.session_state.get(
                f"{grid_key}.columns_state", None
            )

            height = 1000 if len(df) > 20 else 45 * len(df) + 100

            event = st_aggrid.AgGrid(
                df,
                # key=grid_key,
                height=height,
                columns_state=columns_state,
                gridOptions=_build_grid_options(
                    df=df,
                    agg_diff_col=agg_diff_col,
                    diff_cols=diff_cols,
                    record_id_cols=record_id_cols,
                    num_comparators=num_comparators,
                ),
                custom_css={**aggrid_css, **radio_button_css, **diff_cell_css},
                update_on=["selectionChanged"],
                allow_unsafe_jscode=True,
            )
            return pd.DataFrame(event.selected_rows)
        except ImportError:
            # Fallback to st.dataframe if st_aggrid is not installed
            pass

    # Configure column help text for st.dataframe fallback
    column_config = {}
    if num_comparators == 2:
        agg_help = (
            "Average of absolute differences between the two app versions"
        )
        diff_help = "Absolute difference between the two app versions (|version1 - version2|)"
    elif num_comparators > 2:
        agg_help = f"Average standard deviation across all {num_comparators} app versions"
        diff_help = f"Standard deviation of values across all {num_comparators} app versions"

    column_config[agg_diff_col] = st.column_config.NumberColumn(
        help=agg_help, format="%.3f"
    )
    for diff_col in diff_cols:
        column_config[diff_col] = st.column_config.NumberColumn(
            help=diff_help, format="%.3f"
        )

    column_order = ["input", agg_diff_col, *diff_cols]
    column_order = [col for col in column_order if col in df.columns]
    event = st.dataframe(
        df[column_order],
        column_order=column_order,
        column_config=column_config,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
        use_container_width=True,
    )
    return df.iloc[event.selection["rows"]]


def _render_shared_records(
    col_data: Dict[str, Dict[str, pd.DataFrame]],
    feedback_cols: List[str],
):
    query_col = None
    cols = ["input", "record_id"] + feedback_cols
    app_versions = []
    for _, data in col_data.items():
        app_df = data["records"].drop_duplicates(
            subset=["input"] + feedback_cols
        )
        version = data["version"]
        app_versions.append(version)

        if query_col is None:
            query_col = app_df[cols].rename(
                columns=lambda x: f"{x}_{version}"
                if x in cols and x != "input"
                else x
            )
        else:
            query_col = query_col.merge(
                app_df[cols],
                how="inner",
                on="input",
            ).rename(
                columns=lambda x: f"{x}_{version}"
                if x in cols and x != "input"
                else x
            )
    assert query_col is not None

    record_id_cols = [
        f"record_id_{version}"
        for version in app_versions
        if f"record_id_{version}" in query_col.columns
    ]
    # sort query_col columns
    query_col = query_col[sorted(query_col.columns)].set_index("input")

    if query_col.empty:
        st.warning("No shared records found.")
        return

    _render_advanced_filters(query_col, feedback_cols)
    query_col = st.session_state.get(
        f"{page_name}.record_filter.result", query_col
    )
    # Feedback difference
    diff_cols = []
    if len(col_data) == 2:
        col_suffix = "Diff"
    else:
        col_suffix = "Std Dev"
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
            diff_col_name = feedback_col_name + " Std Dev"
            query_col[diff_col_name] = query_col.iloc[
                :, query_col.columns.str.startswith(feedback_col_name)
            ].std(axis=1)

    agg_diff_col = f"Mean {col_suffix}"
    query_col[agg_diff_col] = query_col[diff_cols].mean(axis=1)
    query_col = query_col.sort_values(
        by=[agg_diff_col], ascending=False
    ).reset_index()

    st.subheader("Shared Records Stats")

    selected_rows = _render_grid(
        query_col[["input", agg_diff_col] + diff_cols + record_id_cols],
        agg_diff_col,
        diff_cols,
        record_id_cols,
        len(col_data),
        grid_key="compare_grid",
    )

    if selected_rows is None or selected_rows.empty:
        return None
    return selected_rows[record_id_cols]


def _lookup_app_version(
    versions_df: pd.DataFrame,
    app_version: Optional[str] = None,
    app_id: Optional[str] = None,
):
    if app_version and app_id:
        raise ValueError("Can only pass one of `app_id` or `app_version`")
    try:
        if app_version:
            return versions_df[versions_df["app_version"] == app_version].iloc[
                0
            ]
        elif app_id:
            return versions_df[versions_df["app_id"] == app_id].iloc[0]
        else:
            raise ValueError("Must pass one of `app_id` or `app_version`")
    except IndexError:
        return None


def _render_version_selectors(
    app_name: str,
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

    inc_col, dec_col, _ = st_columns([0.15, 0.15, 0.7])
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

    app_filter_cols = st_columns(
        len(current_app_ids),
        gap="large",
        vertical_alignment="top",
    )

    # Get versions and pinned versions
    versions = versions_df["app_version"].tolist()
    pinned_versions = None
    if PINNED_COL_NAME in versions_df.columns:
        pinned_versions = list(
            versions_df[versions_df[PINNED_COL_NAME]]["app_version"].unique()
        )

    # Determine which versions to show based on pinned versions checkbox
    select_idxs = []
    select_optionss = []
    cur_idx = 0
    for i, app_id in enumerate(current_app_ids):
        select_versions = versions
        checkbox_key = f"{page_name}.app_version_selector_{i}.show_pinned"
        if pinned_versions and app_filter_cols[i].checkbox(
            "Only Show Pinned", key=checkbox_key
        ):
            select_versions = pinned_versions
        if app_id:
            app_row = _lookup_app_version(versions_df, app_id=app_id)
            version = app_row["app_version"] if app_row is not None else None
        if app_id and version in select_versions:
            idx = select_versions.index(version)
        else:
            if cur_idx < len(select_versions):
                idx = cur_idx
                cur_idx += 1
            else:
                idx = 0

        select_optionss.append(select_versions)
        select_idxs.append(idx)

    # Render version selectors
    with st.form("app_version_selector_form", border=False):
        app_selector_cols = st_columns(
            len(current_app_ids),
            gap="large",
            vertical_alignment="top",
        )

        for i, (select_options, select_idx, app_id) in enumerate(
            zip(select_optionss, select_idxs, current_app_ids)
        ):
            selectbox_key = f"{page_name}.app_version_selector_{i}"
            if version := app_selector_cols[i].selectbox(
                f"App Version {i}",
                key=selectbox_key,
                options=select_options,
                index=select_idx,
                format_func=lambda x: f"📌 {x}"
                if pinned_versions and x in pinned_versions
                else x,
            ):
                app_row = _lookup_app_version(versions_df, app_version=version)
                app_id = app_row["app_id"] if app_row is not None else None
                if app_id:
                    current_app_ids[i] = app_id

        if st.form_submit_button("Apply", type="primary"):
            if len(current_app_ids) != len(set(current_app_ids)):
                st.warning("Duplicate app versions selected.")
            st.session_state[f"{page_name}.app_ids"] = current_app_ids
            st.query_params["app_ids"] = ",".join(
                str(app_id) for app_id in current_app_ids
            )

            records, feedback_cols = get_records_and_feedback(
                app_ids=current_app_ids,
                app_name=app_name,
            )
            records = _preprocess_df(records)
            col_data = {
                app_id: {
                    "version": app_df["app_version"].unique()[0],
                    "records": app_df[~app_df[HIDE_RECORD_COL_NAME]]
                    if HIDE_RECORD_COL_NAME in app_df.columns
                    else app_df,
                    "feedback_cols": [
                        col for col in feedback_cols if col in app_df.columns
                    ],
                }
                for app_id, app_df in records.groupby(by="app_id")
            }
            st.session_state[f"{page_name}.col_data"] = col_data
            st.session_state[f"{page_name}.col_data_app_name"] = app_name


def _reset_page_state():
    delete_keys = [
        # f"{page_name}.app_ids",
        f"{page_name}.col_data",
        f"{page_name}.col_data_app_name",
        f"{page_name}.record_filter.result",
    ]
    for key in delete_keys:
        if key in st.session_state:
            del st.session_state[key]


def render_app_comparison(app_name: str):
    """Render the Compare page.

    Args:
        app_name (str): The name of the app to display app versions for comparison.
    """
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
            "Not enough app versions found to compare. Try a different page instead."
        )
        return

    # get app version and record data
    _render_version_selectors(app_name, versions_df)
    if st.session_state.get(f"{page_name}.col_data_app_name", None) != app_name:
        _reset_page_state()
        return
    col_data = st.session_state.get(f"{page_name}.col_data", None)
    if not col_data:
        _reset_page_state()
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
        app_feedback_container.header("App Feedback Comparison")
        _render_all_app_feedback_plot(col_data, feedback_col_names)

    with record_selector_container:
        record_selector_container.header("Overlapping Records")
        selected_record_ids = _render_shared_records(
            col_data, feedback_col_names
        )

    if selected_record_ids is None:
        st.info(
            "Click a record's checkbox to view details.",
            icon="ℹ️",
        )
        return
    record_data = {}
    for app_id, data in col_data.items():
        records = data["records"]
        version = data["version"]
        record_id_col = f"record_id_{version}"
        if record_id_col in selected_record_ids:
            selected = selected_record_ids[record_id_col].iloc[0]
        else:
            st.error("Missing record ID")
        records = records[records["record_id"] == selected]
        record_data[app_id] = {**data, "records": records}
    record_header_container.divider()
    record_header_container.header("Record Comparison")
    with record_feedback_graph_container:
        _render_all_app_feedback_plot(record_data, feedback_col_names)

    record_feedback_selector_container.subheader("Feedback Results")
    with record_feedback_selector_container:
        if selected_ff := _render_feedback_pills(
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
        ):
            feedback_selector_cols = st_columns(
                len(record_data),
                gap="large",
                container=record_feedback_selector_container,
            )
            for i, app_id in enumerate(record_data):
                with feedback_selector_cols[i]:
                    record_df = record_data[app_id]["records"]
                    selected_row = record_df.iloc[0]
                    _render_feedback_call(
                        selected_ff, selected_row, feedback_directions
                    )

    with trace_viewer_container:
        trace_cols = st_columns(
            len(record_data), gap="large", container=trace_viewer_container
        )
        for i, app_id in enumerate(record_data):
            with trace_cols[i]:
                record_df = record_data[app_id]["records"]
                selected_row = record_df.iloc[0]

                record_json = selected_row["record_json"]
                app_json = selected_row["app_json"]

                if is_sis_compatibility_enabled():
                    st.subheader("Trace Details")
                    st.json(record_json, expanded=1)

                    st.subheader("App Details")
                    st.json(app_json, expanded=1)
                elif is_otel_tracing_enabled():
                    event_spans = _get_event_otel_spans(
                        selected_row["record_id"]
                    )
                    if event_spans:
                        record_viewer_otel(
                            spans=event_spans, key=f"compare_{app_id}"
                        )
                    else:
                        st.warning("No trace data available for this record.")
                else:
                    record_viewer(
                        record_json,
                        app_json,
                        key=f"compare_{app_id}",
                    )


def compare_main():
    set_page_config(page_title=page_name)
    init_page_state()
    app_name = render_sidebar()
    if app_name:
        render_app_comparison(app_name)


if __name__ == "__main__":
    compare_main()
