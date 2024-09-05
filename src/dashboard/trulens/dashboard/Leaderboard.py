import math
from typing import Dict, List, Optional, Sequence

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import streamlit as st
from trulens.core.utils.text import format_quantity
from trulens.dashboard.pages.Compare import MAX_COMPARATORS
from trulens.dashboard.pages.Compare import MIN_COMPARATORS
from trulens.dashboard.pages.Compare import page_name as compare_page_name
from trulens.dashboard.pages.Records import page_name as records_page_name
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
from trulens.dashboard.utils.dashboard_utils import update_app_metadata
from trulens.dashboard.ux.components import draw_metadata_and_tags
from trulens.dashboard.ux.styles import CATEGORY
from trulens.dashboard.ux.styles import Category
from trulens.dashboard.ux.styles import cell_rules
from trulens.dashboard.ux.styles import cell_rules_styles
from trulens.dashboard.ux.styles import default_direction
from trulens.dashboard.ux.styles import stmetricdelta_hidearrow

page_name = "Leaderboard"

APP_COLS = ["app_version", "app_id", "app_name"]
APP_AGG_COLS = ["Records", "Average Latency"]


def init_page_state(app_name: str):
    if st.session_state.get(f"{page_name}.initialized", False):
        return

    if app_name:
        add_query_param(ST_APP_NAME, app_name)

    read_query_params_into_session_state(
        page_name=page_name,
        transforms={
            "metadata_to_front": lambda x: x == "True",
            "show_pinned": lambda x: x == "True",
        },
    )
    st.session_state[f"{page_name}.initialized"] = True


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
        "Average Latency": ("latency", "mean"),
        "Total Cost (USD)": ("total_cost_usd", "sum"),
        "Total Cost (Snowflake Credits)": ("total_cost_sf", "sum"),
        "Total Tokens": ("total_tokens", "sum"),
        "tags": ("tags", lambda x: ",".join(x.drop_duplicates())),
    }
    for col in feedback_col_names:
        agg_dict[col] = (col, "mean")

    app_agg_df: pd.DataFrame = (
        records_df.groupby(
            by=["app_version", "app_name", "app_id"], dropna=True, sort=True
        )
        .aggregate(**agg_dict)
        .reset_index()
    )

    if "_leaderboard.pinned" in app_versions_df:
        app_versions_df["_leaderboard.pinned"] = app_versions_df[
            "_leaderboard.pinned"
        ].astype(bool)

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
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_default_column(resizable=True)
    gb.configure_column(
        "app_version",
        header_name="App Version",
        resizable=True,
        pinned="left",
    )

    gb.configure_column(
        "_leaderboard.pinned",
        header_name="Leaderboard",
        hide=True,
    )
    gb.configure_column(
        "app_id",
        header_name="App ID",
        hide=True,
        resizable=True,
    )
    gb.configure_column(
        "app_name",
        header_name="App Name",
        hide=True,
        resizable=True,
    )

    for feedback_col in feedback_col_names:
        if "distance" in feedback_col:
            gb.configure_column(
                feedback_col, hide=feedback_col.endswith("_calls")
            )
        else:
            # cell highlight depending on feedback direction
            feedback_direction = (
                "HIGHER_IS_BETTER"
                if feedback_directions.get(feedback_col, default_direction)
                else "LOWER_IS_BETTER"
            )

            gb.configure_column(
                feedback_col,
                cellClassRules=cell_rules[feedback_direction],
                hide=feedback_col.endswith("_calls"),
            )

    gb.configure_grid_options(rowHeight=40)
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True, paginationPageSize=25)
    gb.configure_side_bar()
    gb.configure_grid_options(
        autoSizeStrategy={"type": "fitCellContents", "skipHeader": False}
    )
    return gb.build()


# @st.fragment
def _render_grid(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
    grid_key: Optional[str] = None,
):
    return AgGrid(
        df,
        key=grid_key,
        height=600,
        gridOptions=_build_grid_options(
            df=df,
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
            version_metadata_col_names=version_metadata_col_names,
        ),
        custom_css=cell_rules_styles,
        update_on=["selectionChanged"],
        allow_unsafe_jscode=True,
    )


@st.fragment
def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    grid_key: Optional[str] = None,
):
    container = st.container()
    c1, c2, c3, c4 = container.columns(
        [1, 1, 1, 1], gap="large", vertical_alignment="center"
    )
    if metadata_to_front := c1.toggle(
        "Metadata to Front",
        key=f"{page_name}.metadata_to_front",
    ):
        df = order_columns(
            df,
            APP_COLS
            + version_metadata_col_names
            + APP_AGG_COLS
            + feedback_col_names,
        )
    else:
        df = order_columns(
            df,
            APP_COLS
            + APP_AGG_COLS
            + feedback_col_names
            + version_metadata_col_names,
        )
    st.query_params["metadata_to_front"] = str(metadata_to_front)

    if show_pinned := c1.toggle(
        "Show Pinned",
        key=f"{page_name}.show_pinned",
    ):
        if "_leaderboard.pinned" in df:
            df = df[df["_leaderboard.pinned"]]
        else:
            st.info(
                "Pin an app version by selecting it and clicking the `Pin` button.",
                icon="📌",
            )
    st.query_params["show_pinned"] = str(show_pinned)

    grid_data = _render_grid(
        df,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
        version_metadata_col_names=version_metadata_col_names,
        grid_key=grid_key,
    )
    selected_rows = grid_data.selected_rows
    selected_rows = pd.DataFrame(selected_rows)

    if selected_rows.empty:
        st.info("Click an App Version's checkbox to view details.")
        selected_app_ids = []
    else:
        selected_app_ids = list(selected_rows.app_id.unique())
        st.dataframe(selected_rows.set_index("app_id"))

    # Add to Leaderboard
    on_leaderboard = any(
        "_leaderboard.pinned" in app and app["_leaderboard.pinned"]
        for _, app in selected_rows.iterrows()
    )
    if c2.button(
        "Unpin" if on_leaderboard else "Pin",
        key=f"{grid_key}_pin_button",
        disabled=selected_rows.empty,
    ):
        for app_id in selected_app_ids:
            update_app_metadata(
                app_id, {"_leaderboard": {"pinned": not on_leaderboard}}
            )
        st.cache_data.clear()
        if on_leaderboard:
            st.toast(
                f"Successfully removed {len(selected_app_ids)} app(s) from Leaderboard"
            )
        else:
            st.toast(
                f"Successfully added {len(selected_app_ids)} app(s) to Leaderboard"
            )
        st.rerun()

    # Examine Records
    if c3.button(
        "Examine Records",
        disabled=selected_rows.empty,
        use_container_width=True,
    ):
        st.session_state[f"{records_page_name}.app_ids"] = selected_app_ids
        st.switch_page("pages/Records.py")
    # Compare App Versions
    if len(selected_app_ids) < MIN_COMPARATORS:
        _compare_button_label = f"Min {MIN_COMPARATORS} Apps"
        _compare_button_disabled = True
    elif len(selected_app_ids) > MAX_COMPARATORS:
        _compare_button_label = f"Max {MAX_COMPARATORS} Apps"
        _compare_button_disabled = True
    else:
        _compare_button_label = "Compare"
        _compare_button_disabled = False
    if c4.button(
        _compare_button_label,
        disabled=_compare_button_disabled,
        use_container_width=True,
    ):
        st.session_state[f"{compare_page_name}.app_ids"] = selected_app_ids
        st.switch_page("pages/Compare.py")


@st.fragment
def _render_list_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    max_feedback_cols: int = 5,
):
    st.markdown(
        stmetricdelta_hidearrow,
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
            f"#### {app_version}", help=draw_metadata_and_tags(metadata, tags)
        )
        st.caption(f"App ID: {app_id}")
        app_feedback_col_names = [
            col_name
            for col_name in feedback_col_names
            if col_name in app_row and app_row[col_name] is not None
        ]
        (
            n_records_col,
            latency_col,
            cost_col,
            tokens_col,
            select_app_col,
        ) = st.columns([1, 1, 1, 1, 1])
        feedback_cols = st.columns(
            min(len(app_feedback_col_names), max_feedback_cols)
        )

        n_records_col.metric("Records", app_row["Records"])

        latency_mean = app_row["Average Latency"]
        latency_col.metric(
            "Average Latency (Seconds)",
            (
                f"{format_quantity(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean)
                else "nan"
            ),
        )

        if app_row["Total Cost (Snowflake Credits)"] > 0:
            cost_col.metric(
                "Total Cost (Snowflake credits)",
                f"{format_quantity(round(app_row['Total Cost (Snowflake Credits)'], 8), precision=5)}",
            )
        elif app_row["Total Cost (USD)"] > 0:
            cost_col.metric(
                "Total Cost (USD)",
                f"${format_quantity(round(app_row['Total Cost (USD)'], 5), precision=2)}",
            )

        tokens_col.metric(
            "Total Tokens",
            format_quantity(
                app_row["Total Tokens"],
                precision=2,
            ),
        )

        col_counter = 0
        for col_name in app_feedback_col_names:
            mean = app_row[col_name]
            if mean is None or pd.isna(mean):
                continue
            col = feedback_cols[col_counter % max_feedback_cols]
            col_counter += 1
            feedback_container = col.container(border=True)

            higher_is_better = feedback_directions.get(col_name, True)

            if "distance" in col_name:
                feedback_container.metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal",
                )
            else:
                cat: Category = CATEGORY.of_score(
                    mean, higher_is_better=higher_is_better
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
                            mean, CATEGORY.PASS[cat.direction].threshold
                        )
                        else "inverse"
                    ),
                )

        with select_app_col:
            if st.button(
                "Select App",
                key=f"select_app_{app_id}",
            ):
                st.session_state["Records_app_id"] = app_id
                st.switch_page("pages/Records.py")

        # with st.expander("Model metadata"):
        #    st.markdown(draw_metadata(metadata))

        st.markdown("""---""")


def render_leaderboard(app_name: str):
    st.title(page_name)
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, version_metadata_col_names = render_app_version_filters(
        app_name
    )
    st.divider()

    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_df, feedback_col_names = get_records_and_feedback(
        app_ids, limit=1000
    )
    if records_df.empty:
        st.error(f"No records found for app `{app_name}`.")
        return
    elif len(records_df) == 1000:
        st.info(
            "Computed using the last 1000 records.",
            icon="ℹ️",
        )

    feedback_col_names = list(feedback_col_names)
    # Preprocess data
    df = _preprocess_df(
        records_df,
        versions_df,
        list(feedback_col_names),
        version_metadata_col_names,
    )
    _, feedback_directions = get_feedback_defs()

    versions_tab, list_tab = st.tabs([
        "App Versions",
        "List",
    ])
    with versions_tab:
        _render_grid_tab(
            df,
            grid_key="leaderboard_grid",
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
            version_metadata_col_names=version_metadata_col_names,
        )
    with list_tab:
        _render_list_tab(
            df,
            feedback_col_names,
            feedback_directions,
            version_metadata_col_names,
        )


if __name__ == "__main__":
    set_page_config(page_title=page_name)
    app_name = render_sidebar()
    if app_name:
        init_page_state(app_name)
        render_leaderboard(app_name)
