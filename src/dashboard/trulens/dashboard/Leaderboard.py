import asyncio
import json
import math

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from trulens.core import TruSession
from trulens.core.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens.core.utils.text import format_quantity
from trulens.dashboard.streamlit_utils import init_from_args
from trulens.dashboard.ux import styles
from trulens.dashboard.ux.components import draw_metadata_and_tags
from trulens.dashboard.ux.page_config import set_page_config
from trulens.dashboard.ux.styles import CATEGORY

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

if __name__ == "__main__":
    # If not imported, gets args from command line and creates Tru singleton
    init_from_args()


def leaderboard():
    """Render the leaderboard page."""

    set_page_config(page_title="Leaderboard")

    session = TruSession()  # get singleton whether this file was imported or executed from command line.

    lms = session.connector.db

    # Set the title
    st.title("App Leaderboard")

    def get_data():
        return lms.get_records_and_feedback([])

    def get_apps():
        return list(lms.get_apps())

    # sort apps by name
    def sort_selected_apps(selected_apps, records):
        """
        Sorts the selected_apps list based on the concatenation of app_name and app_version.

        Parameters:
        selected_apps (list): List of app_ids to be sorted.
        apps (DataFrame): DataFrame containing app_id and app_json columns.

        Returns:
        list: Sorted list of selected_apps.
        """
        # Create a mapping from app_id to (app_name, app_version)
        app_info = (
            records.set_index("app_id")["app_json"]
            .apply(
                lambda x: (
                    json.loads(x).get("app_name"),
                    json.loads(x).get("app_version"),
                )
            )
            .to_dict()
        )

        # Sort selected_apps based on the concatenation of app_name and app_version
        sorted_apps = sorted(
            selected_apps,
            key=lambda app_id: f"{app_info[app_id][0]}{app_info[app_id][1]}",
        )

        return sorted_apps

    records, feedback_col_names = get_data()
    records = records.sort_values(by="app_id")

    apps = get_apps()
    app_names = sorted(list(set(app["app_name"] for app in apps)))

    selected_app_names = st.multiselect("Filter apps:", app_names, app_names)
    selected_apps = [
        app["app_id"]
        for app in apps
        if any(name in app["app_name"] for name in selected_app_names)
    ]
    st.session_state.app = selected_apps

    # Filter app versions to only include those from selected apps
    filtered_apps = [app for app in apps if app["app_id"] in selected_apps]
    app_name_versions = sorted(
        list(
            set(
                f"{app['app_name']} - {app['app_version']}"
                for app in filtered_apps
            )
        )
    )

    selected_app_name_versions = st.multiselect(
        "Filter by app version:", app_name_versions, app_name_versions
    )
    selected_apps = [
        app["app_id"]
        for app in apps
        if f"{app['app_name']} - {app['app_version']}"
        in selected_app_name_versions
    ]
    st.session_state.app = selected_apps

    with st.expander("Advanced Filters"):
        # get tag options
        tags = []
        for i in range(len(apps)):
            tags.append(apps[i]["tags"])
        unique_tags = sorted(list(set(tags)))
        # select tags
        selected_tags = st.multiselect("Filter tags:", unique_tags, unique_tags)

        # filter to apps with selected tags
        tag_selected_apps = [
            app["app_id"]
            for app in apps
            if any(tag in app["tags"] for tag in selected_tags)
        ]
        selected_apps = list(set(selected_apps) & set(tag_selected_apps))
        st.session_state.app = selected_apps

        # get metadata options
        metadata_keys_unique = set()
        for app in apps:
            metadata_keys_unique.update(app["metadata"].keys())
        metadata_keys_unique = list(metadata_keys_unique)
        # select metadata
        metadata_options = {}
        for metadata_key in metadata_keys_unique:
            unique_values = set()
            for i in range(len(apps)):
                unique_values.add(apps[i]["metadata"][metadata_key])
            metadata_options[metadata_key] = list(unique_values)

        # select metadata
        metadata_selections = metadata_options.copy()
        for metadata_key in metadata_options.keys():
            metadata_selections[metadata_key] = st.multiselect(
                "Filter " + metadata_key + ":",
                sorted(metadata_options[metadata_key]),
                sorted(metadata_options[metadata_key]),
            )

        # filter to apps with selected metadata
        metadata_selected_apps = [
            app["app_id"]
            for app in apps
            if all(
                app["metadata"][metadata_key]
                in metadata_selections[metadata_key]
                for metadata_key in metadata_selections.keys()
            )
        ]

        selected_apps = list(set(selected_apps) & set(metadata_selected_apps))
        st.session_state.app = selected_apps

    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "")
            or row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }

    if records.empty:
        st.write("No records yet...")
        return

    if records.empty:
        st.write("No records yet...")

    st.markdown("""---""")

    selected_apps = sort_selected_apps(selected_apps, records)

    for app in selected_apps:
        app_df = records.loc[records.app_id == app]
        if app_df.empty:
            continue
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        app_name = app_json["app_name"]
        app_version = app_json["app_version"]
        app_name_version = f"{app_name} - {app_version}"
        metadata = app_json.get("metadata")
        tags = app_json.get("tags")
        # st.text('Metadata' + str(metadata))
        st.header(app_name_version, help=draw_metadata_and_tags(metadata, tags))
        app_feedback_col_names = [
            col_name
            for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols, col99 = st.columns(
            5 + len(app_feedback_col_names)
        )
        latency_mean = (
            app_df["latency"]
            .apply(lambda td: td if td != MIGRATION_UNKNOWN_STR else None)
            .mean()
        )

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            (
                f"{format_quantity(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean)
                else "nan"
            ),
        )
        col3.metric(
            "Total Cost (USD)",
            f"${format_quantity(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision=2)}",
        )
        col4.metric(
            "Total Tokens",
            format_quantity(
                sum(
                    tokens
                    for tokens in app_df.total_tokens
                    if tokens is not None
                ),
                precision=2,
            ),
        )

        for i, col_name in enumerate(app_feedback_col_names):
            mean = app_df[col_name].mean()

            st.write(
                styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            higher_is_better = feedback_directions.get(col_name, True)

            if "distance" in col_name:
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal",
                )
            else:
                cat = CATEGORY.of_score(mean, higher_is_better=higher_is_better)
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal"
                        if cat.compare(
                            mean, CATEGORY.PASS[cat.direction].threshold
                        )
                        else "inverse"
                    ),
                )

        with col99:
            if st.button("Select App", key=f"app-selector-{app}"):
                st.session_state.app = app
                switch_page("Evaluations")

        # with st.expander("Model metadata"):
        #    st.markdown(draw_metadata(metadata))

        st.markdown("""---""")


if __name__ == "__main__":
    leaderboard()
