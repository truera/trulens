import os
from typing import List, Optional

import pandas as pd
import streamlit as st
from trulens.dashboard.constants import STABILITY_PAGE_NAME as page_name
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config


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

    records_df, feedback_col_names = get_records_and_feedback(app_ids=app_ids)

    if "QA_ACCURACY" not in feedback_col_names:
        st.warning(
            "QA_ACCURACY feedback function not found. "
            "Please ensure you have QA_ACCURACY feedback defined."
        )
        return pd.DataFrame()

    ret = records_df[["app_version", "input_id", "input", "QA_ACCURACY"]]

    def unique_input(series):
        if series.nunique() > 1:
            raise ValueError(
                "Not all values in the 'input' column are the same for this group."
            )
        return series.iloc[0]

    ret = (
        ret.groupby(["app_version", "input_id"])
        .agg(Input=("input", unique_input), QA_ACCURACY=("QA_ACCURACY", list))
        .reset_index()
    )
    ret.rename(
        columns={"app_version": "App Version", "input_id": "Input ID"},
        inplace=True,
    )
    ret["Stability"] = ret["QA_ACCURACY"].apply(
        lambda x: x.count(1) / len(x) if len(x) > 0 else 0
    )
    ret["Total Records"] = ret["QA_ACCURACY"].apply(len)
    return ret


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
    st.dataframe(stability_df)


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
