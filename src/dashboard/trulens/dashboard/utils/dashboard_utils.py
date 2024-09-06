import argparse
from datetime import datetime
from datetime import timedelta
import sys
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st
from trulens.core import __package__ as core_package
from trulens.core import __version__ as core_version
from trulens.core.database import base as mod_db
from trulens.core.session import TruSession
from trulens.core.utils.imports import static_resource
from trulens.dashboard import __package__ as dashboard_package
from trulens.dashboard import __version__
from trulens.dashboard.constants import CACHE_TTL
from trulens.dashboard.constants import PINNED_COL_NAME

ST_APP_NAME = "app_name"
ST_APP_VERSION = "app_version"
ST_APP_ID = "app_id"


def set_page_config(page_title: Optional[str] = None):
    page_title = f"TruLens: {page_title}" if page_title else "TruLens"
    st.set_page_config(
        page_title=page_title,
        page_icon="https://www.trulens.org/img/favicon.ico",
        layout="wide",
    )

    if st.get_option("theme.base") == "dark":
        logo = str(static_resource("dashboard", "ux/trulens_logo_light.svg"))
        logo_small = str(
            static_resource("dashboard", "ux/trulens_squid_light.svg")
        )
    else:
        logo = str(static_resource("dashboard", "ux/trulens_logo.svg"))
        logo_small = str(static_resource("dashboard", "ux/trulens_squid.svg"))

    st.logo(logo, icon_image=logo_small, link="https://www.trulens.org/")


def add_query_param(param_name: str, param_value: str):
    st.query_params[param_name] = param_value


def read_query_params_into_session_state(
    page_name: str, transforms: Optional[dict[str, Callable[[str], Any]]] = None
):
    """This method loads query params into the session state. This function should only be called only once when the page is first initialized.

    Args:
        page_name (str): Name of the page being initialized. Used to prefix page-specific session keys.
        transforms (Optional[dict[str, Callable]], optional): An optional dictionary mapping query param names to a function that deserializes the respective query arg value. Defaults to None.

    """
    assert not st.session_state.get(f"{page_name}.initialized", False)
    for param, value in st.query_params.to_dict().items():
        prefix_page_name = True
        if param.startswith("filter."):
            prefix_page_name = False
            if param.endswith(".multiselect"):
                value = value.split(",")
                if len(value) == 1 and value[0] == "":
                    value = []
        elif transforms and param in transforms:
            value = transforms[param](value)

        if prefix_page_name:
            st.session_state[f"{page_name}.{param}"] = value
        else:
            st.session_state[param] = value


@st.cache_resource(show_spinner="Setting up TruLens session")
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

    st.session_state["cache.last_refreshed"] = datetime.now()
    return TruSession(
        database_url=args.database_url, database_prefix=args.database_prefix
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting record data")
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


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting app data")
def get_apps():
    session = get_session()
    lms = session.connector.db
    assert lms
    return list(lms.get_apps())


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting feedback definitions")
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
    app_name = None

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
            st.session_state["cache.last_refreshed"] = datetime.now()
            st.rerun()
        if "cache.last_refreshed" in st.session_state:
            last_refreshed: datetime = st.session_state["cache.last_refreshed"]
            tdelta: timedelta = datetime.now() - last_refreshed
            if tdelta.seconds < 5 * 60:
                last_refreshed_str = "just now"
            elif tdelta.seconds < 60 * 60:
                last_refreshed_str = f"{tdelta.seconds // 60} minutes ago"
            elif tdelta.days == 0:
                last_refreshed_str = last_refreshed.strftime("%H:%M:%S")
            else:
                last_refreshed_str = last_refreshed.strftime("%m-%d-%Y")

            st.sidebar.text(f"Last refreshed {last_refreshed_str}")

    with st.sidebar.expander("Info"):
        st.text(f"{core_package}\nv{core_version}")
        st.text(f"{dashboard_package}\nv{__version__}")

        st.link_button(
            "Share Feedback",
            "https://forms.gle/HAc4HBk5nZRpgw7C6",
            help="Help us improve TruLens!",
        )
    if app_name is None:
        st.error("No apps found in the database.")
    return app_name


def _flatten_metadata(metadata: dict):
    results = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            for k2, v2 in _flatten_metadata(v).items():
                results[f"{k}.{k2}"] = v2
        else:
            results[k] = str(v)
    return results


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting app versions")
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
    if PINNED_COL_NAME in app_versions_df.columns:
        app_versions_df[PINNED_COL_NAME] = app_versions_df[
            PINNED_COL_NAME
        ].astype(bool)
    return app_versions_df, list(app_version_metadata_cols)


def _get_query_args_handler(key: str, max_options: Optional[int] = None):
    new_val = st.session_state.get(key)
    if isinstance(new_val, list):
        if len(new_val) == max_options:
            # don't need to explicitly add query args as default is all options
            del st.query_params[key]
            return
        new_val = ",".join(str(v) for v in new_val)
    elif not isinstance(new_val, str):
        raise ValueError(
            f"Unable to save value to query params: {new_val} (type: {type(new_val)})"
        )
    st.query_params[key] = new_val


def _render_filter_multiselect(name: str, options: List[str], key: str):
    return st.multiselect(
        name,
        options,
        default=options,
        key=key,
        on_change=_get_query_args_handler,
        args=(
            key,
            len(options),
        ),
    )


def _handle_reset_filters(
    keys: List[str],
    tags: List[str],
    metadata_options: Dict[str, List[str]],
):
    for key in keys:
        if key == "filter.search":
            val = ""
        elif key == "filter.tags.multiselect":
            val = tags
        elif key.startswith("filter.metadata.") and key.endswith(
            ".multiselect"
        ):
            metadata_key = key[16:-12]
            val = metadata_options[metadata_key]
        else:
            raise ValueError(f"Invalid key found: {key}")
        st.session_state[key] = val
        del st.query_params[key]


def render_app_version_filters(app_name: str):
    app_versions_df, app_version_metadata_cols = get_app_versions(app_name)
    filtered_app_versions = app_versions_df

    col0, col1, col2 = st.columns(
        [0.7, 0.15, 0.15], vertical_alignment="bottom"
    )
    active_adv_filters = []
    if version_str_query := col0.text_input(
        "Search App Version",
        key="filter.search",
        on_change=_get_query_args_handler,
        args=("filter.search",),
    ):
        active_adv_filters.append("filter.search")
        filtered_app_versions = filtered_app_versions[
            filtered_app_versions["app_version"].str.contains(
                version_str_query, case=False
            )
        ]

    with col1.popover("Advanced Filters", use_container_width=True):
        # get tag options
        st.header("Advanced Filters")
        tags = set()
        for _, app_version in app_versions_df.iterrows():
            tags |= set(app_version["tags"])
        tags = sorted(tags)
        # select tags

        selected_tags = _render_filter_multiselect(
            "tags", tags, key="filter.tags.multiselect"
        )
        if len(selected_tags) != len(tags):
            active_adv_filters.append("filter.tags.multiselect")

        metadata_options = {}
        for metadata_key in app_version_metadata_cols:
            try:
                unique_values = app_versions_df[metadata_key].unique()
            except TypeError:
                pass
            if len(unique_values):
                metadata_options[metadata_key] = list(unique_values)

        # select metadata
        metadata_selections = {}
        for metadata_key in metadata_options.keys():
            _metadata_select_options = sorted(
                metadata_options[metadata_key], key=str
            )
            metadata_selections[metadata_key] = _render_filter_multiselect(
                metadata_key,
                _metadata_select_options,
                key=f"filter.metadata.{metadata_key}.multiselect",
            )
            if len(metadata_selections[metadata_key]) != len(
                _metadata_select_options
            ):
                active_adv_filters.append(
                    f"filter.metadata.{metadata_key}.multiselect"
                )

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

    if len(active_adv_filters):
        col2.button(
            "Reset Filters",
            use_container_width=True,
            type="primary",
            on_click=_handle_reset_filters,
            args=(active_adv_filters, tags, metadata_options),
        )

    return filtered_app_versions, app_version_metadata_cols
