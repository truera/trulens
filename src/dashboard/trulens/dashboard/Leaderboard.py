import asyncio
import json
import math

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from trulens.core import Tru
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

    tru = Tru()  # get singletone whether this file was imported or executed from command line.

    lms = tru.db

    # Set the title
    st.title("App Leaderboard")

    # wrapper so we can cache the records and feedback
    @st.cache_data
    def get_data():
        return lms.get_records_and_feedback([])
    
    @st.cache_data
    def get_apps():
        return list(lms.get_apps())
    
    records, feedback_col_names = get_data()
    records = records.sort_values(by="app_id")

    apps = get_apps()
    app_ids = []
    for i in range(len(apps)):
        app_ids.append(apps[i]['app_id'])

    selected_apps = st.multiselect("Filter apps:", app_ids, app_ids)

    with st.expander("Advanced Filters"):
        # get tag options
        tags = []
        for i in range(len(apps)):
            tags.append(apps[i]['tags'])
        unique_tags = list(set(tags))
        # select tags
        selected_tags = st.multiselect("Filter tags:", unique_tags, unique_tags)

        # filter to apps with selected tags
        tag_selected_apps = [app['app_id'] for app in apps if any(tag in app['tags'] for tag in selected_tags)]
        selected_apps = list(set(selected_apps) & set(tag_selected_apps))

        # get metadata options
        metadata_keys_unique = set()
        for app in apps:
            metadata_keys_unique.update(app['metadata'].keys())
        metadata_keys_unique = list(metadata_keys_unique)
        # select metadata
        metadata_options = {}
        for metadata_key in metadata_keys_unique:
            unique_values = set()
            for i in range(len(apps)):
                unique_values.add(apps[i]['metadata'][metadata_key])
            metadata_options[metadata_key] = list(unique_values)

        # select metadata
        metadata_selections = metadata_options.copy()
        for metadata_key in metadata_options.keys():
            metadata_selections[metadata_key] = st.multiselect("Filter " + metadata_key + ":", metadata_options[metadata_key], metadata_options[metadata_key])
        
        # sort apps by name
        selected_apps = sorted(selected_apps)


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

    for app in selected_apps:
        app_df = records.loc[records.app_id == app]
        if app_df.empty:
            continue
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        metadata = app_json.get("metadata")
        tags = app_json.get("tags")
        # st.text('Metadata' + str(metadata))
        st.header(app, help=draw_metadata_and_tags(metadata, tags))
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
