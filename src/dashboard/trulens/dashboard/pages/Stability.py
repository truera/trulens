import os
from typing import List, Optional

import pandas as pd
import streamlit as st
from trulens.dashboard.constants import STABILITY_PAGE_NAME as page_name
from trulens.dashboard.utils.dashboard_utils import _parse_json_fields
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux.styles import aggrid_css
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


def _get_stability_data(app_ids: List[str]) -> pd.DataFrame:
    """Get stability data grouped by input_id with QA_ACCURACY metrics."""
    session = get_session()
    db = session.connector.db

    if not db or not hasattr(db, "get_events"):
        st.error("Database must support OTEL spans for Stability analysis")
        return pd.DataFrame()

    try:
        # Get all events for the app_ids
        events_df = db.get_events(
            app_name=None, app_version=None, record_ids=None, start_time=None
        )

        if events_df.empty:
            return pd.DataFrame()

        # Filter events that have the input_id attribute and are for our app_ids
        stability_data = []

        for _, event in events_df.iterrows():
            record_attributes = _parse_json_fields(
                event.get("record_attributes", {})
            )

            if (
                isinstance(record_attributes, dict)
                and "error" not in record_attributes
            ):
                input_id = record_attributes.get("ai.observability.input_id")
                record_id = record_attributes.get("ai.observability.record_id")
                app_id = record_attributes.get(
                    "ai.observability.recording.app_id"
                )

                if input_id and record_id and app_id in app_ids:
                    stability_data.append({
                        "input_id": input_id,
                        "record_id": record_id,
                        "app_id": app_id,
                    })

        if not stability_data:
            return pd.DataFrame()

        stability_df = pd.DataFrame(stability_data)

        # Get records and feedback to get QA_ACCURACY scores
        records_df, feedback_col_names = get_records_and_feedback(
            app_ids=app_ids
        )

        if "QA_ACCURACY" not in feedback_col_names:
            st.warning(
                "QA_ACCURACY feedback function not found. "
                "Please ensure you have QA_ACCURACY feedback defined."
            )
            # Check if there are any feedback columns with "accuracy" in the name
            accuracy_cols = [
                col for col in feedback_col_names if "accuracy" in col.lower()
            ]
            if accuracy_cols:
                st.info(
                    f"Found these accuracy-related feedback columns: "
                    f"{', '.join(accuracy_cols)}"
                )
                st.info(
                    "You can modify the code to use one of these columns "
                    "instead of QA_ACCURACY"
                )
            return pd.DataFrame()

        # Merge stability data with records to get QA_ACCURACY scores
        merged_df = stability_df.merge(
            records_df[
                ["record_id", "QA_ACCURACY", "input", "output", "app_version"]
            ],
            on="record_id",
            how="left",
        )

        # Group by input_id and calculate stability metrics
        grouped = (
            merged_df.groupby("input_id")
            .agg({
                "record_id": "count",
                "QA_ACCURACY": lambda x: (x == 1).sum() / len(x)
                if len(x) > 0
                else 0,
                "input": "first",  # Take the first input for display
                "output": lambda x: list(x),  # Keep all outputs for inspection
                "app_version": lambda x: list(
                    x.unique()
                ),  # Keep unique app versions
            })
            .reset_index()
        )

        grouped.columns = [
            "Input ID",
            "Total Records",
            "Stability",
            "Input Text",
            "Outputs",
            "App Versions",
        ]
        grouped["Stability"] = grouped["Stability"].round(3)

        return grouped

    except Exception as e:
        st.error(f"Error getting stability data: {e}")
        return pd.DataFrame()


def _render_stability_grid(df: pd.DataFrame):
    """Render the stability data grid."""
    if df.empty:
        st.warning("No stability data found for the selected app versions.")
        return None

    # Display columns for the grid (excluding Outputs which might be large)
    display_df = df[
        ["Input ID", "Total Records", "Stability", "Input Text", "App Versions"]
    ].copy()

    try:
        import st_aggrid
        from st_aggrid.shared import ColumnsAutoSizeMode
        from st_aggrid.shared import DataReturnMode
        from st_aggrid.shared import GridOptionsBuilder

        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_selection(
            "single", use_checkbox=False, rowMultiSelectWithClick=False
        )
        gb.configure_column(
            "Stability",
            type=["numericColumn", "numberColumnFilter"],
            precision=3,
        )
        gb.configure_column(
            "Total Records", type=["numericColumn", "numberColumnFilter"]
        )
        gb.configure_column(
            "Input Text", wrapText=True, autoHeight=True, maxWidth=400
        )
        gb.configure_column(
            "App Versions", wrapText=True, autoHeight=True, maxWidth=200
        )

        gridOptions = gb.build()

        event = st_aggrid.AgGrid(
            display_df,
            gridOptions=gridOptions,
            update_on=["selectionChanged"],
            custom_css={**aggrid_css, **radio_button_css},
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            allow_unsafe_jscode=True,
        )

        selected_rows = pd.DataFrame(event.selected_rows)
        if not selected_rows.empty:
            selected_input_id = selected_rows.iloc[0]["Input ID"]
            return selected_input_id

    except ImportError:
        # Fallback to st.dataframe if st_aggrid is not installed
        event = st.dataframe(
            display_df,
            selection_mode="single-row",
            on_select="rerun",
            hide_index=True,
            use_container_width=True,
        )
        if event.selection["rows"]:
            selected_idx = event.selection["rows"][0]
            selected_input_id = display_df.iloc[selected_idx]["Input ID"]
            return selected_input_id

    return None


def _render_input_details(df: pd.DataFrame, selected_input_id: str):
    """Render details for the selected input_id."""
    selected_row = df[df["Input ID"] == selected_input_id].iloc[0]

    st.subheader(f"Details for Input ID: {selected_input_id}")

    col1, col2, col3 = st_columns(3)
    with col1:
        st.metric("Total Records", selected_row["Total Records"])
    with col2:
        st.metric("Stability Score", f"{selected_row['Stability']:.3f}")
    with col3:
        stability_pct = selected_row["Stability"] * 100
        st.metric("Success Rate", f"{stability_pct:.1f}%")

    st.subheader("Input Text")
    st.text_area(
        "", value=selected_row["Input Text"], height=100, disabled=True
    )

    st.subheader("App Versions")
    app_versions = selected_row["App Versions"]
    if isinstance(app_versions, list):
        for version in app_versions:
            st.write(f"• {version}")
    else:
        st.write(app_versions)

    st.subheader("All Outputs")
    outputs = selected_row["Outputs"]
    if isinstance(outputs, list):
        for i, output in enumerate(outputs, 1):
            with st.expander(f"Output {i}"):
                st.text(output)
    else:
        st.text(outputs)


def render_stability(app_name: str):
    """Render the stability analysis page."""
    st.title(page_name)
    st.markdown(f"Showing stability analysis for app `{app_name}`")
    st.markdown(
        "This page shows the stability of different inputs by grouping "
        "records with the same `ai.observability.input_id` and calculating "
        "the proportion that have a `QA_ACCURACY` score of 1."
    )

    # Get app versions
    versions_df, _ = render_app_version_filters(app_name, {}, page_name_keys=[])

    st.divider()

    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return

    app_ids = versions_df["app_id"].tolist()

    # Get stability data
    with st.spinner("Analyzing stability data..."):
        stability_df = _get_stability_data(app_ids)

    if stability_df.empty:
        st.warning(
            "No stability data found. Make sure your records have "
            "`ai.observability.input_id` span attributes and `QA_ACCURACY` "
            "feedback."
        )
        return

    # Render the grid
    selected_input_id = _render_stability_grid(stability_df)

    # Show details if an input is selected
    if selected_input_id:
        st.divider()
        _render_input_details(stability_df, selected_input_id)


def stability_main():
    """Main function for the stability page."""
    # Check if the stability tab should be shown
    if os.getenv("TRULENS_STABILITY_TAB") not in ["1", "true"]:
        st.error(
            "Stability tab is not enabled. Set TRULENS_STABILITY_TAB=1 or "
            "TRULENS_STABILITY_TAB=true to enable."
        )
        return

    set_page_config(page_title=page_name)
    init_page_state()
    app_name = render_sidebar()
    if app_name:
        render_stability(app_name)


if __name__ == "__main__":
    stability_main()
