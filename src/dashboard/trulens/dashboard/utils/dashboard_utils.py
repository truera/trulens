import argparse
import sys
from typing import List, Optional

import pandas as pd
import streamlit as st
from trulens.core import __package__ as core_package
from trulens.core import __version__ as core_version
from trulens.core.database import base as mod_db
from trulens.core.session import TruSession
from trulens.core.utils.imports import static_resource
from trulens.dashboard import __package__ as dashboard_package
from trulens.dashboard import __version__

ST_APP_NAME = "app_name"
ST_APP_VERSION = "app_version"
ST_APP_ID = "app_id"


def set_page_config(page_title: Optional[str] = None):
    page_title = f"TruLens: {page_title}" if page_title else "TruLens"
    st.set_page_config(
        page_title=f"TruLens: {page_title}",
        page_icon="https://www.trulens.org/img/favicon.ico",
        layout="wide",
    )

    logo = static_resource("dashboard", "ux/trulens_logo.svg")

    st.logo(str(logo), link="https://www.trulens.org/")


def read_query_params_into_session_state(page_name: str = "global"):
    initialized = st.session_state.get(f"{page_name}.initialized", False)
    if not initialized:
        for param, value in st.query_params.to_dict().items():
            print(param, value, page_name)
            st.session_state[f"{page_name}.{param}"] = value
        st.session_state[f"{page_name}.initialized"] = True


@st.cache_resource
def get_session() -> TruSession:
    """Parse command line arguments and initialize TruSession with them.

    As TruSession is a singleton, further TruSession() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--database-prefix", default=mod_db.DEFAULT_DATABASE_PREFIX
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    return TruSession(
        database_url=args.database_url, database_prefix=args.database_prefix
    )


@st.cache_data
def get_records_and_feedback(
    app_ids: Optional[List[str]] = None, limit: Optional[int] = None
):
    session = get_session()
    lms = session.connector.db
    assert lms
    records_df, feedback_col_names = lms.get_records_and_feedback(
        app_ids=app_ids, limit=limit
    )
    records_df = records_df.replace({float("nan"): None})
    return records_df, feedback_col_names


@st.cache_data
def get_apps():
    session = get_session()
    lms = session.connector.db
    assert lms
    return list(lms.get_apps())


@st.cache_data
def get_feedback_defs():
    session = get_session()
    lms = session.connector.db
    assert lms

    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "")
            or row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }
    return feedback_defs, feedback_directions


def update_app_metadata(app_id: str, metadata: dict):
    session = get_session()
    lms = session.connector.db
    assert lms
    lms.update_app_metadata(app_id, metadata)


def render_sidebar():
    apps = get_apps()

    if apps:
        app_names = sorted(list(set(app["app_name"] for app in apps)))

        if len(app_names) > 1:
            app_name = st.sidebar.selectbox(
                "Select an app", options=app_names, disabled=len(app_names) == 1
            )
        else:
            app_name = app_names[0]

        if app_name and app_name != st.session_state.get(ST_APP_NAME):
            st.session_state[ST_APP_NAME] = app_name

        if st.sidebar.button("↻ Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.sidebar.expander("Info"):
        st.text(f"{core_package}\nv{core_version}")
        st.text(f"{dashboard_package}\nv{__version__}")

        st.link_button(
            "Share Feedback",
            "https://forms.gle/HAc4HBk5nZRpgw7C6",
            help="Help us improve TruLens!",
        )


def _flatten_metadata(metadata: dict):
    results = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            for k2, v2 in _flatten_metadata(v).items():
                results[f"{k}.{k2}"] = v2
        else:
            results[k] = v
    return results


@st.cache_data
def get_app_versions(app_name: str):
    apps = get_apps()
    app_versions = [app for app in apps if app["app_name"] == app_name]
    app_versions_df = pd.DataFrame(app_versions)

    # Flatten metadata
    app_versions_df["metadata"] = app_versions_df["metadata"].apply(
        lambda x: _flatten_metadata(x) if isinstance(x, dict) else {}
    )

    # Factor out metadata
    app_version_metadata_cols = set()
    for _, app in app_versions_df.iterrows():
        app_version_metadata_cols.update(app["metadata"].keys())

    for metadata_key in app_version_metadata_cols:
        app_versions_df[metadata_key] = app_versions_df["metadata"].apply(
            lambda x: x.get(metadata_key, None)
        )
    app_versions_df = app_versions_df.replace({float("nan"): None})
    return app_versions_df, list(app_version_metadata_cols)


def render_app_version_filters(app_name: str):
    app_versions_df, app_version_metadata_cols = get_app_versions(app_name)
    filtered_app_versions = app_versions_df

    col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")
    if version_str_query := col1.text_input("Search App Version", key="search"):
        filtered_app_versions = filtered_app_versions[
            filtered_app_versions["app_version"].str.contains(
                version_str_query, case=False
            )
        ]

    with col2.popover("Advanced Filters", use_container_width=True):
        # get tag options
        st.header("Advanced Filters")
        tags = set()
        for _, app_version in app_versions_df.iterrows():
            tags |= set(app_version["tags"])
        tags = sorted(tags)
        # select tags
        selected_tags = st.multiselect("tags", tags, tags)

        metadata_options = {}
        for metadata_key in app_version_metadata_cols:
            try:
                unique_values = app_versions_df[metadata_key].unique()
            except TypeError:
                pass
            if len(unique_values):
                metadata_options[metadata_key] = list(unique_values)

        # select metadata
        metadata_selections = metadata_options.copy()
        for metadata_key in metadata_options.keys():
            metadata_selections[metadata_key] = st.multiselect(
                metadata_key,
                sorted(metadata_options[metadata_key], key=str),
                sorted(metadata_options[metadata_key], key=str),
            )

        # submitted = st.form_submit_button("Apply", type="primary")
        # if submitted:

        # filter to apps with selected metadata
        for metadata_key in metadata_selections.keys():
            filtered_app_versions = filtered_app_versions[
                filtered_app_versions[metadata_key].isin(
                    metadata_selections[metadata_key]
                )
            ]
        filtered_app_versions = filtered_app_versions[
            filtered_app_versions["tags"].apply(
                lambda x: any(tag in x for tag in selected_tags)
            )
        ]
    # st.markdown(
    #     f":blue-background[Got {len(filtered_app_versions)} App Versions]"
    # )
    return filtered_app_versions, app_version_metadata_cols
