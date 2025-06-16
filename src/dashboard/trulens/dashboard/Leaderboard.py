import math
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from trulens.apps import virtual as virtual_app
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import text as text_utils
from trulens.dashboard import constants as dashboard_constants
from trulens.dashboard.pages import Compare as Compare_page
from trulens.dashboard.utils import dashboard_utils
from trulens.dashboard.utils import metadata_utils
from trulens.dashboard.utils import streamlit_compat
from trulens.dashboard.utils.dashboard_utils import _show_no_records_error
from trulens.dashboard.utils.dashboard_utils import is_sis_compatibility_enabled
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux import components as dashboard_components
from trulens.dashboard.ux import styles as dashboard_styles

APP_COLS = ["app_version", "app_id", "app_name"]
APP_AGG_COLS = ["Records", "Average Latency (s)"]


def init_page_state():
    if st.session_state.get(
        f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.initialized", False
    ):
        return

    dashboard_utils.read_query_params_into_session_state(
        page_name=dashboard_constants.LEADERBOARD_PAGE_NAME,
        transforms={
            "only_show_pinned": lambda x: x == "True",
            "metadata_cols": lambda x: x.split(","),
        },
    )
    st.session_state[
        f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.initialized"
    ] = True


def _preprocess_df(
    records_df: pd.DataFrame,
    app_versions_df: pd.DataFrame,
    feedback_col_names: List[str],
    metadata_col_names: List[str],
    show_all: bool = False,
):
    records_df = records_df.sort_values(by="app_id")

    records_df["total_cost_usd"] = records_df["total_cost"].where(
        records_df["cost_currency"] == "USD",
        other=0,
    )
    records_df["total_cost_sf"] = records_df["total_cost"].where(
        records_df["cost_currency"] == "Snowflake credits",
        other=0,
    )

    agg_dict = {
        "Records": ("record_id", "count"),
        "Average Latency (s)": ("latency", "mean"),
        "Total Cost (USD)": ("total_cost_usd", "sum"),
        "Total Cost (Snowflake Credits)": ("total_cost_sf", "sum"),
        "Total Tokens": ("total_tokens", "sum"),
        "tags": ("tags", lambda x: ",".join(x.drop_duplicates())),
    }
    for col in feedback_col_names:
        if col in records_df:
            agg_dict[col] = (col, "mean")

    app_agg_df: pd.DataFrame = (
        records_df.groupby(
            by=["app_version", "app_name", "app_id"], dropna=True, sort=True
        )
        .aggregate(**agg_dict)
        .reset_index()
    )

    df = app_agg_df.join(
        app_versions_df.set_index(["app_id", "app_name", "app_version"])[
            metadata_col_names
        ],
        how="left" if not show_all else "right",
        validate="many_to_one",
        on=["app_id", "app_name", "app_version"],
    ).round(3)
    return df


def order_columns(
    df: pd.DataFrame,
    order: Sequence[str],
):
    return df[order]


def _build_grid_options(
    df: pd.DataFrame,
    feedback_col_names: List[str],
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
        filter="agMultiColumnFilter",
    )

    gb.configure_columns(APP_COLS, filter="agMultiColumnFilter")
    gb.configure_columns(
        feedback_col_names + ["records", "latency"],
        filter="agNumberColumnFilter",
    )
    gb.configure_columns(
        version_metadata_col_names, filter="agMultiColumnFilter", editable=True
    )

    gb.configure_column(
        dashboard_constants.PINNED_COL_NAME,
        header_name="Pinned",
        hide=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        dashboard_constants.EXTERNAL_APP_COL_NAME,
        header_name="External App",
        hide=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "app_id",
        header_name="App ID",
        hide=True,
        resizable=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "app_name",
        header_name="App Name",
        hide=True,
        resizable=True,
        filter="agMultiColumnFilter",
    )

    for feedback_col in feedback_col_names:
        if "distance" in feedback_col:
            gb.configure_column(
                feedback_col,
                hide=feedback_col.endswith("_calls"),
                filter="agNumberColumnFilter",
            )
        else:
            # cell highlight depending on feedback direction
            feedback_direction = (
                "HIGHER_IS_BETTER"
                if feedback_directions.get(
                    feedback_col, dashboard_styles.default_direction
                )
                else "LOWER_IS_BETTER"
            )

            gb.configure_column(
                feedback_col,
                cellClassRules=dashboard_styles.cell_rules[feedback_direction],
                hide=feedback_col.endswith("_calls"),
                filter="agNumberColumnFilter",
            )

    gb.configure_grid_options(
        rowHeight=45,
        suppressContextMenu=True,
        rowClassRules={
            # "external-app": f"data['{mod_constants.EXTERNAL_APP_COL_NAME}'] > 0",
            "app-external": f"data['{dashboard_constants.EXTERNAL_APP_COL_NAME}']",
            "app-pinned": f"data['{dashboard_constants.PINNED_COL_NAME}']",
        },
    )
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True, paginationPageSize=25)
    gb.configure_side_bar()
    gb.configure_grid_options(autoSizeStrategy={"type": "fitGridWidth"})
    return gb.build()


def _render_grid(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    grid_key: Optional[str] = None,
):
    if not is_sis_compatibility_enabled():
        try:
            import st_aggrid

            columns_state = st.session_state.get(
                f"{grid_key}.columns_state", None
            )

            if dashboard_constants.PINNED_COL_NAME in df:
                df.loc[
                    df[dashboard_constants.PINNED_COL_NAME], "app_version"
                ] = df.loc[
                    df[dashboard_constants.PINNED_COL_NAME], "app_version"
                ].apply(lambda x: f"📌 {x}")

            event = st_aggrid.AgGrid(
                df,
                key=grid_key,
                columns_state=columns_state,
                gridOptions=_build_grid_options(
                    df=df,
                    feedback_col_names=feedback_col_names,
                    feedback_directions=feedback_directions,
                    version_metadata_col_names=version_metadata_col_names,
                ),
                custom_css=dashboard_styles.aggrid_css,
                update_on=["selectionChanged", "cellValueChanged"],
                allow_unsafe_jscode=True,
            )

            if (
                event.event_data
                and event.event_data["type"] == "cellValueChanged"
            ):
                handle_table_edit(
                    df, event.event_data, version_metadata_col_names
                )
            return pd.DataFrame(event.selected_rows)
        except ImportError:
            # Fallback to st.dataframe if st_aggrid is not installed
            pass

    column_order = [
        "app_version",
        "records",
        "latency",
        *feedback_col_names,
    ]
    column_order = [col for col in column_order if col in df.columns]
    event = st.dataframe(
        df[column_order],
        column_order=column_order,
        selection_mode="multi-row",
        on_select="rerun",
        hide_index=True,
        use_container_width=True,
    )
    return df.iloc[event.selection["rows"]]


def handle_pin_toggle(selected_app_ids: List[str], on_leaderboard: bool):
    # Create nested metadata dict
    value = metadata_utils.nest_metadata(
        dashboard_constants.PINNED_COL_NAME, not on_leaderboard
    )
    for app_id in selected_app_ids:
        dashboard_utils.update_app_metadata(app_id, value)
    dashboard_utils.get_app_versions.clear()
    dashboard_utils.get_apps.clear()
    if on_leaderboard:
        st.toast(
            f"Successfully removed {len(selected_app_ids)} app(s) from Leaderboard"
        )
    else:
        st.toast(
            f"Successfully added {len(selected_app_ids)} app(s) to Leaderboard"
        )


def handle_table_edit(
    df: pd.DataFrame,
    event_data: Dict[str, Any],
    version_metadata_col_names: List[str],
):
    app_id = event_data["data"]["app_id"]
    if app_id not in df["app_id"].values:
        st.error(f"App with ID {app_id} not found in the leaderboard!")
        return

    app_df = df[df["app_id"] == app_id].iloc[0]
    metadata = {}
    for col in version_metadata_col_names:
        if col in event_data["data"] and event_data["data"][col] != app_df[col]:
            value = metadata_utils.nest_metadata(col, event_data["data"][col])
            metadata_utils.nested_update(metadata, value)
    dashboard_utils.update_app_metadata(app_id, metadata)

    dashboard_utils.get_app_versions.clear()
    dashboard_utils.get_apps.clear()
    st.toast(f"Successfully updated metadata for `{app_df['app_version']}`")


@streamlit_compat.st_dialog("Add/Edit Metadata")
def handle_add_metadata(
    selected_rows: pd.DataFrame, metadata_col_names: List[str]
):
    st.write(
        f"Add or edit metadata for {len(selected_rows)} selected app version(s)"
    )
    key = st.text_input("Metadata Key", placeholder="metadata.key")

    if key and not re.match(r"^[A-Za-z0-9_.]+$", key):
        st.error(
            "`key` must contain only alphanumeric and underscore characters!"
        )

    existing_value = None
    placeholder = None
    if key in selected_rows.columns:
        existing_values = list(selected_rows[key].unique())
        if len(existing_values) == 1:
            existing_value = existing_values[0]
        else:
            placeholder = "<multiple existing values>"

    val = st.text_input(
        "Metadata Value",
        placeholder=placeholder or "value",
        value=existing_value,
    )

    if st.button("Submit"):
        if not key:
            st.error("Metadata key cannot be empty!")
            return
        metadata = metadata_utils.nest_metadata(key, val)
        for app_id in selected_rows["app_id"]:
            dashboard_utils.update_app_metadata(app_id, metadata)

        if key not in metadata_col_names:
            metadata_col_names.append(key)

        dashboard_utils.get_app_versions.clear()
        dashboard_utils.get_apps.clear()
        st.toast(
            f"Successfully updated metadata for {len(selected_rows)} app(s)"
        )
        st.rerun()


@streamlit_compat.st_dialog("Add Virtual App")
def handle_add_virtual_app(
    app_name: str,
    feedback_col_names: List[str],
    feedback_defs: Any,
    metadata_col_names: List[str],
):
    with st.form(
        f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.add_virtual_app_form",
        border=False,
    ):
        app_version = st.text_input(
            "App Version", placeholder="virtual_app_base"
        )

        feedback_values = {
            col: st.number_input(col)
            for col in feedback_col_names
            if col in feedback_defs["feedback_name"].values
        }
        metadata_values = metadata_utils.nest_dict({
            col: st.text_input(col)
            for col in metadata_col_names
            if not col.startswith("trulens.")
        })

        metadata_values = {k: v for k, v in metadata_values.items() if v}
        metadata_values[dashboard_constants.EXTERNAL_APP_COL_NAME] = True

        if st.form_submit_button("Submit"):
            if app_version and not re.match(r"^[A-Za-z0-9_.]+$", app_version):
                st.error(
                    "`app_version` must contain only alphanumeric and underscore characters!"
                )
                return
            session = dashboard_utils.get_session()
            app = virtual_app.TruVirtual(
                app=virtual_app.VirtualApp(),
                app_name=app_name,
                app_version=app_version,
                metadata=metadata_values,
            )

            virtual_record = virtual_app.VirtualRecord(
                calls={},
                main_input="<autogenerated record>",
                main_output="<autogenerated record>",
                meta=metadata_utils.nest_metadata(
                    dashboard_constants.HIDE_RECORD_COL_NAME, True
                ),
            )

            app.add_record(virtual_record)
            for feedback_name, feedback_value in feedback_values.items():
                result = feedback_schema.FeedbackResult(
                    record_id=virtual_record.record_id,
                    feedback_definition_id=feedback_defs[
                        feedback_defs["feedback_name"] == feedback_name
                    ]["feedback_definition_id"].iloc[0],
                    status=feedback_schema.FeedbackResultStatus.DONE,
                    cost={},
                    perf={},
                    name=feedback_name,
                    result=feedback_value,
                )
                session.connector.db.insert_feedback(result)

            dashboard_utils.get_records_and_feedback.clear()
            dashboard_utils.get_app_versions.clear()
            dashboard_utils.get_apps.clear()
            st.toast(f"Successfully created virtual app version {app_version}.")
            st.rerun()


def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_defs: Any,
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    app_name: str,
    grid_key: Optional[str] = None,
):
    container = st.container()
    c1, c2, c3, c4, c5, c6 = st_columns(
        [1, 1, 1, 1, 1, 1], vertical_alignment="center", container=container
    )

    _metadata_options = [
        col
        for col in version_metadata_col_names
        if not col.startswith("trulens.")
    ]

    # Validate metadata_cols
    if metadata_col_values := st.session_state.get(
        f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.metadata_cols", []
    ):
        metadata_select_options = [
            col_name
            for col_name in metadata_col_values
            if col_name in _metadata_options
        ]
    else:
        metadata_select_options = _metadata_options

    metadata_cols = st.multiselect(
        label="Display Metadata Columns",
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.metadata_cols",
        options=metadata_select_options,
        default=metadata_select_options,
    )
    if len(metadata_cols) != len(_metadata_options):
        st.query_params["metadata_cols"] = ",".join(metadata_cols)
    if (
        dashboard_constants.EXTERNAL_APP_COL_NAME in df
        and dashboard_constants.EXTERNAL_APP_COL_NAME not in metadata_cols
    ):
        metadata_cols.append(dashboard_constants.EXTERNAL_APP_COL_NAME)
    if (
        dashboard_constants.PINNED_COL_NAME in df
        and dashboard_constants.PINNED_COL_NAME not in metadata_cols
    ):
        metadata_cols.append(dashboard_constants.PINNED_COL_NAME)

    df = order_columns(
        df,
        APP_COLS + APP_AGG_COLS + feedback_col_names + metadata_cols,
    )

    if only_show_pinned := c1.toggle(
        "Only Show Pinned",
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.only_show_pinned",
    ):
        if dashboard_constants.PINNED_COL_NAME in df:
            df = df[df[dashboard_constants.PINNED_COL_NAME]]
        else:
            st.info(
                "Pin an app version by selecting it and clicking the `Pin App` button.",
                icon="📌",
            )
            return
    st.query_params["only_show_pinned"] = str(only_show_pinned)

    selected_rows = _render_grid(
        df,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
        version_metadata_col_names=version_metadata_col_names,
        grid_key=grid_key,
    )

    if selected_rows is None or selected_rows.empty:
        selected_app_ids = []
    else:
        selected_app_ids = list(selected_rows.app_id.unique())

    # Add to Leaderboard
    on_leaderboard = any(
        dashboard_constants.PINNED_COL_NAME in app
        and app[dashboard_constants.PINNED_COL_NAME]
        for _, app in selected_rows.iterrows()
    )
    if c2.button(
        "Unpin App" if on_leaderboard else "Pin App",
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.pin_button",
        disabled=selected_rows.empty,
        on_click=handle_pin_toggle,
        use_container_width=True,
        args=(selected_app_ids, on_leaderboard),
    ):
        st.rerun()
    # Examine Records
    if c3.button(
        "Examine Records",
        disabled=selected_rows.empty,
        use_container_width=True,
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.records_button",
    ):
        st.session_state[f"{dashboard_constants.RECORDS_PAGE_NAME}.app_ids"] = (
            selected_app_ids
        )
        st.switch_page("pages/Records.py")
    # Compare App Versions
    if len(selected_app_ids) < Compare_page.MIN_COMPARATORS:
        _compare_button_disabled = True
        help_msg = f"Select at least {Compare_page.MIN_COMPARATORS} app versions to compare."
    elif len(selected_app_ids) > Compare_page.MAX_COMPARATORS:
        _compare_button_disabled = True
        help_msg = f"Deselect to at most {Compare_page.MAX_COMPARATORS} app versions to compare."
    else:
        _compare_button_disabled = False
        help_msg = None

    _compare_button_label = "Compare"

    if c4.button(
        _compare_button_label,
        help=help_msg,
        disabled=_compare_button_disabled,
        use_container_width=True,
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.compare_button",
    ):
        st.session_state[f"{dashboard_constants.COMPARE_PAGE_NAME}.app_ids"] = (
            selected_app_ids
        )
        st.switch_page("pages/Compare.py")

    # Add Metadata Col
    if c5.button(
        "Add/Edit Metadata",
        disabled=selected_rows.empty,
        use_container_width=True,
        key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.modify_metadata_button",
    ):
        handle_add_metadata(selected_rows, version_metadata_col_names)

    # Virtual apps do not work in OTEL world.
    if not is_otel_tracing_enabled():
        # Add Virtual App
        if c6.button(
            "Add Virtual App",
            use_container_width=True,
            key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.add_virtual_app_button",
        ):
            handle_add_virtual_app(
                app_name,
                feedback_col_names,
                feedback_defs,
                version_metadata_col_names,
            )


@streamlit_compat.st_fragment
def _render_list_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    max_feedback_cols: int = 6,
):
    st.markdown(
        dashboard_styles.stmetricdelta_hidearrow,
        unsafe_allow_html=True,
    )
    for _, app_row in df.iterrows():
        app_id = app_row["app_id"]
        app_version = app_row["app_version"]
        tags = app_row["tags"]
        metadata = {
            col: app_row[col]
            for col in version_metadata_col_names
            if col in app_row
        }
        st.markdown(
            f"#### {app_version}",
            help=dashboard_components.draw_metadata_and_tags(metadata, tags),
        )
        st.caption(app_id)
        app_feedback_col_names = [
            col_name
            for col_name in feedback_col_names
            if col_name in app_row and app_row[col_name] is not None
        ]
        (
            n_records_col,
            latency_col,
            tokens_col,
            cost_col,
            select_app_col,
        ) = st_columns([1, 1, 1, 1, 1])
        n_records_col.metric("Records", app_row["Records"])

        latency_mean = app_row["Average Latency (s)"]
        latency_col.metric(
            "Average Latency (Seconds)",
            (
                f"{text_utils.format_quantity(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean)
                else "nan"
            ),
        )

        if app_row["Total Cost (Snowflake Credits)"] > 0:
            cost_col.metric(
                "Total Cost (Snowflake credits)",
                f"{text_utils.format_quantity(round(app_row['Total Cost (Snowflake Credits)'], 8), precision=5)}",
            )
        elif app_row["Total Cost (USD)"] > 0:
            cost_col.metric(
                "Total Cost (USD)",
                f"${text_utils.format_quantity(round(app_row['Total Cost (USD)'], 5), precision=2)}",
            )

        tokens_col.metric(
            "Total Tokens",
            text_utils.format_quantity(
                app_row["Total Tokens"],
                precision=2,
            ),
        )

        if len(app_feedback_col_names) > 0:
            feedback_cols = st_columns(
                min(len(app_feedback_col_names), max_feedback_cols)
            )
            for i, col_name in enumerate(app_feedback_col_names):
                mean = app_row[col_name]
                if mean is None or pd.isna(mean):
                    continue
                col = feedback_cols[i % max_feedback_cols]
                feedback_container = col.container(border=True)

                higher_is_better = feedback_directions.get(col_name, True)

                if "distance" in col_name:
                    feedback_container.metric(
                        label=col_name,
                        value=f"{round(mean, 2)}",
                        delta_color="normal",
                    )
                else:
                    cat: dashboard_styles.Category = (
                        dashboard_styles.CATEGORY.of_score(
                            mean, higher_is_better=higher_is_better
                        )
                    )
                    feedback_container.metric(
                        label=col_name,
                        value=f"{round(mean, 2)}",
                        delta=f"{cat.icon} {cat.adjective}",
                        delta_color=(
                            "normal"
                            if cat.compare
                            and cat.direction
                            and cat.compare(
                                mean,
                                dashboard_styles.CATEGORY.PASS[
                                    cat.direction
                                ].threshold,
                            )
                            else "inverse"
                        ),
                    )

        with select_app_col:
            if st.button(
                "Select App Version",
                key=f"select_app_version_{app_id}",
            ):
                st.session_state[
                    f"{dashboard_constants.RECORDS_PAGE_NAME}.app_ids"
                ] = [app_id]
                st.switch_page("pages/Records.py")

        st.markdown("""---""")


@streamlit_compat.st_fragment
def _render_plot_tab(df: pd.DataFrame, feedback_col_names: List[str]):
    if len(feedback_col_names) == 0:
        st.warning("No feedback functions found.")
        return
    if dashboard_constants.HIDE_RECORD_COL_NAME in df.columns:
        df = df[~df[dashboard_constants.HIDE_RECORD_COL_NAME]]
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
                "size": 0.1,
            },
            texttemplate="%{y}",
            name="",  # Stops trace {i} from showing up in popup annotation.
        )
        fig.add_trace(
            plot,
            row=row_num,
            col=col_num,
        )
        if i == 0:
            xaxis = fig["layout"]["xaxis"]
            yaxis = fig["layout"]["yaxis"]
        else:
            xaxis = fig["layout"][f"xaxis{i + 1}"]
            yaxis = fig["layout"][f"yaxis{i + 1}"]

        xaxis["title"] = "Score"
        if col_num == 1:
            yaxis["title"] = "# Records"

    fig.update_layout(
        height=300 * rows,
        width=200 * cols,
        dragmode=False,
        showlegend=False,
        barcornerradius=4,
        bargap=0.05,
    )
    fig.update_yaxes(fixedrange=True, showgrid=False)
    # Histogram bins are [start_inclusive, end_exclusive), so extend the range
    # by the step to the right.
    fig.update_xaxes(
        fixedrange=True, showgrid=False, autorangeoptions={"include": [0, 1]}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_leaderboard(app_name: str):
    """Renders the Leaderboard page.

    Args:
        app_name (str): The app name to render the leaderboard for.
    """
    st.title(dashboard_constants.LEADERBOARD_PAGE_NAME)
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, version_metadata_col_names = (
        dashboard_utils.render_app_version_filters(app_name)
    )
    st.divider()

    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_limit = st.session_state.get(dashboard_utils.ST_RECORDS_LIMIT, None)
    records_df, feedback_col_names = dashboard_utils.get_records_and_feedback(
        app_name=app_name, app_ids=app_ids, limit=records_limit
    )
    if records_df.empty:
        # Check for cross-format records before showing generic error
        _show_no_records_error(app_name=app_name, app_ids=app_ids)
        return
    elif records_limit is not None and len(records_df) >= records_limit:
        cols = st_columns([0.9, 0.1], vertical_alignment="center")
        cols[0].info(
            f"Computed from the last {records_limit} records.",
            icon="ℹ️",
        )

        def handle_show_all():
            st.session_state[dashboard_utils.ST_RECORDS_LIMIT] = None
            if dashboard_utils.ST_RECORDS_LIMIT in st.query_params:
                del st.query_params[dashboard_utils.ST_RECORDS_LIMIT]

        cols[1].button(
            "Show all",
            use_container_width=True,
            on_click=handle_show_all,
            help="Show all records. This may take a while.",
        )

    feedback_col_names = list(feedback_col_names)
    # Preprocess data
    df = _preprocess_df(
        records_df,
        versions_df,
        list(feedback_col_names),
        version_metadata_col_names,
    )
    feedback_defs, feedback_directions = dashboard_utils.get_feedback_defs()

    (
        versions_tab,
        plot_tab,
        list_tab,
    ) = st.tabs([
        "App Versions",
        "Feedback Histograms",
        "List View",
    ])
    with versions_tab:
        _render_grid_tab(
            df,
            grid_key=f"{dashboard_constants.LEADERBOARD_PAGE_NAME}.leaderboard_grid",
            feedback_col_names=feedback_col_names,
            feedback_defs=feedback_defs,
            feedback_directions=feedback_directions,
            app_name=app_name,
            version_metadata_col_names=version_metadata_col_names,
        )
    with plot_tab:
        _render_plot_tab(records_df, feedback_col_names)
    with list_tab:
        _render_list_tab(
            df,
            feedback_col_names,
            feedback_directions,
            version_metadata_col_names,
        )


def leaderboard_main():
    dashboard_utils.set_page_config(
        page_title=dashboard_constants.LEADERBOARD_PAGE_NAME
    )
    init_page_state()
    app_name = dashboard_utils.render_sidebar()
    if app_name:
        render_leaderboard(app_name)


if __name__ == "__main__":
    leaderboard_main()
