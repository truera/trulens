import asyncio
from typing import Dict, Sequence, Tuple

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import ColumnsAutoSizeMode
from st_aggrid.shared import DataReturnMode
from st_aggrid.shared import JsCode
from st_keyup import st_keyup
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from trulens.core import Tru
from trulens.dashboard.streamlit_utils import init_from_args
from trulens.dashboard.ux.page_config import set_page_config
from trulens.dashboard.ux.styles import cellstyle_jscode

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())


def _preprocess_df(df: pd.DataFrame, feedback_col_names: Sequence[str]):
    df = df.sort_values(by="app_id")
    agg_dict = {
        "Records": ("record_id", "count"),
        "Average Latency": ("latency", "mean"),
        "Total Cost": ("total_cost", "sum"),
        "Total Tokens": ("total_tokens", "sum"),
    }
    for col in feedback_col_names:
        agg_dict[col] = (col, "mean")

    return (
        df.groupby("app_id", dropna=True, sort=True)
        .agg(**agg_dict)
        .reset_index()
        .round(2)
    )


@st.cache_data
def load_apps_data() -> Tuple[pd.DataFrame, Sequence[str], Dict[str, bool]]:
    tru = Tru()
    lms = tru.db
    df, feedback_col_names = lms.get_records_and_feedback()
    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "")
            or row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }
    df = _preprocess_df(df, feedback_col_names)
    return df, feedback_col_names, feedback_directions


def build_grid_options(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=False)
    # gb.configure_first_column_as_index()
    gb.configure_column(
        "app_id",
        header_name="App ID",
        resizable=True,
        checkboxSelection=True,
    )

    for feedback_col in feedback_col_names:
        if "distance" in feedback_col:
            gb.configure_column(
                feedback_col, hide=feedback_col.endswith("_calls")
            )
        else:
            default_direction = "HIGHER_IS_BETTER"
            # cell highlight depending on feedback direction
            cellstyle = JsCode(
                cellstyle_jscode[
                    "HIGHER_IS_BETTER"
                    if feedback_directions.get(feedback_col, default_direction)
                    else "LOWER_IS_BETTER"
                ]
            )

            gb.configure_column(
                feedback_col,
                cellStyle=cellstyle,
                hide=feedback_col.endswith("_calls"),
            )

    # gb.configure_grid_options(rowHeight=60)
    # gb.configure_auto_height()
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=False)
    # gb.configure_grid_options(
    #     onSelectionChanged=JsCode("""function(params) {
    #             const selectedRows = params.api.getSelectedNodes();
    #             let selectedRowsCount = selectedRows.length;
    #             while (selectedRowsCount > 2) {
    #                 params.api.deselectNode(selectedRows[selectedRowsCount - 1])
    #                 selectedRowsCount--;
    #             }
    #         }"""),
    # )
    gb.configure_side_bar()
    gb = gb.build()
    # st.write(gb)
    return gb


def leaderboard():
    """Render the leaderboard page."""

    set_page_config(page_title="Leaderboard")

    # Set the title and subtitle of the app
    st.title("App Leaderboard")
    st.write(
        "Average feedback values displayed in the range from 0 (worst) to 1 (best)."
    )

    grid_df, feedback_col_names, feedback_directions = load_apps_data()
    if grid_df.empty:
        st.write("No Applications yet...")
        return

    app_filter = st_keyup("App Filter")
    st.write(app_filter)
    if app_filter:
        grid_df = grid_df[grid_df["app_id"].str.contains(app_filter)]

    data = AgGrid(
        grid_df,
        # theme="quartz",
        gridOptions=build_grid_options(
            df=grid_df,
            feedback_col_names=feedback_col_names,
            feedback_directions=feedback_directions,
        ),
        update_on=["selectionChanged"],
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
    )

    selected_rows = data.selected_rows
    selected_rows = pd.DataFrame(selected_rows)

    if selected_rows.empty:
        st.write("No Apps selected")
        return

    # st.markdown("""---""")
    # st.header("Selected Versions")
    # st.write(selected_rows[["app_id"]].reset_index(drop=True))
    col1, col2 = st.columns(2)

    apps = list(selected_rows.app_id.unique())
    if col1.button("Examine Records", type="primary", key="examine"):
        st.session_state["app_ids"] = apps
        switch_page("Records")
    if len(apps) >= 2:
        if col2.button(
            "Compare",
            key="sxs",
            args=(apps,),
        ):
            st.session_state["app_ids"] = apps[:2]
            switch_page("compare")


if __name__ == "__main__":
    # If not imported, gets args from command line and creates Tru singleton
    init_from_args()
    leaderboard()
