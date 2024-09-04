from typing import Dict, List, Optional, Sequence

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import streamlit as st
from trulens.dashboard.utils.dashboard_utils import ST_APP_NAME
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.dashboard_utils import update_app_metadata
from trulens.dashboard.ux.styles import cell_rules
from trulens.dashboard.ux.styles import cell_rules_styles
from trulens.dashboard.ux.styles import default_direction

set_page_config(page_title="Leaderboard")
render_sidebar()
app_name = st.session_state[ST_APP_NAME]

APP_COLS = ["app_version", "app_id", "app_name"]
APP_AGG_COLS = ["Records", "Average Latency"]


def _preprocess_df(
    records_df: pd.DataFrame,
    app_versions_df: pd.DataFrame,
    feedback_col_names: List[str],
    metadata_col_names: List[str],
):
    records_df = records_df.sort_values(by="app_id")
    agg_dict = {
        "Records": ("record_id", "count"),
        "Average Latency": ("latency", "mean"),
        "Total Cost": ("total_cost", "sum"),
        "Total Tokens": ("total_tokens", "sum"),
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


# @st.fragment
def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
    grid_key: Optional[str] = None,
):
    container = st.container()
    c0, c1, c2, c3, c4 = container.columns([3, 1, 1, 1, 1], gap="large")
    if metadata_to_front := c1.toggle(
        "Metadata to Front",
        key=f"{grid_key}_metadata_toggle",
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
        key=f"{grid_key}_pinned_toggle",
    ):
        df = df[df["_leaderboard.pinned"]]
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
        st.write("No Apps selected")
        selected_app_ids = []
    else:
        selected_app_ids = list(selected_rows.app_id.unique())
        c0.write(selected_rows)

    # Examine Records
    if c2.button(
        "Examine Records",
        disabled=selected_rows.empty,
        key=f"{grid_key}_examine",
    ):
        st.switch_page("records")

    # Add to Leaderboard
    on_leaderboard = any(
        "_leaderboard.pinned" in app and app["_leaderboard.pinned"]
        for _, app in selected_rows.iterrows()
    )
    if c3.button(
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

    # Compare App Versions
    if c4.button(
        "Compare" if len(selected_app_ids) == 2 else "Select 2 Apps to Compare",
        key=f"{grid_key}_sxs",
        args=(selected_app_ids,),
        disabled=len(selected_app_ids) != 2,
    ):
        st.switch_page("compare")


@st.fragment
def _render_list_tab(df: pd.DataFrame):
    pass


def render_leaderboard():
    st.title("Leaderboard")
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    versions_df, version_metadata_col_names = render_app_version_filters(
        app_name
    )
    st.divider()

    if versions_df.empty:
        st.write("No versions available for this app.")
        return
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_df, feedback_col_names = get_records_and_feedback(
        app_ids, limit=1000
    )
    if records_df.empty:
        st.write("No data available for this app.")
        return
    elif len(records_df) == 1000:
        st.info(
            "Computed using the last 1000 records.",
            icon="🚨",
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
        _render_list_tab(df)


if __name__ == "__main__":
    render_leaderboard()
