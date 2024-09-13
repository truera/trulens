import argparse
import json
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
from trulens.dashboard.constants import EXTERNAL_APP_COL_NAME
from trulens.dashboard.constants import HIDE_RECORD_COL_NAME
from trulens.dashboard.constants import PINNED_COL_NAME
from trulens.dashboard.constants import RECORDS_LIMIT
from trulens.dashboard.utils.metadata_utils import flatten_metadata

ST_APP_NAME = "app_name"
ST_RECORDS_LIMIT = "records_limit"


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

    if ST_RECORDS_LIMIT not in st.session_state:
        st.session_state[ST_RECORDS_LIMIT] = RECORDS_LIMIT


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
        if param == ST_APP_NAME:
            prefix_page_name = False
        elif param.startswith("filter."):
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

    return TruSession(
        database_url=args.database_url, database_prefix=args.database_prefix
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting record data")
def get_records_and_feedback(
    app_name: str,
    app_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
):
    session = get_session()
    lms = session.connector.db
    assert lms
    records_df, feedback_col_names = lms.get_records_and_feedback(
        app_name=app_name, app_ids=app_ids, limit=limit
    )

    record_json = records_df["record_json"].apply(json.loads)
    records_df["record_metadata"] = record_json.apply(
        lambda x: flatten_metadata(x["meta"])
        if isinstance(x["meta"], dict)
        else {}
    )

    records_df, _ = _factor_out_metadata(records_df, "record_metadata")

    if HIDE_RECORD_COL_NAME in records_df.columns:
        records_df[HIDE_RECORD_COL_NAME] = (
            records_df[HIDE_RECORD_COL_NAME] == "True"
        ).astype(bool)
    records_df = records_df.replace({float("nan"): None})
    return records_df, feedback_col_names


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting app data")
def get_apps(app_name: Optional[str] = None):
    session = get_session()
    lms = session.connector.db
    assert lms
    return list(lms.get_apps(app_name=app_name))


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
    feedback_defs["feedback_name"] = feedback_defs["feedback_json"].map(
        lambda x: x.get("supplied_name", "")
    )
    return feedback_defs, feedback_directions


def update_app_metadata(app_id: str, metadata: dict):
    session = get_session()
    lms = session.connector.db
    assert lms
    lms.update_app_metadata(app_id, metadata)


def _handle_app_selection(app_names: List[str]):
    value = st.session_state.get(f"{ST_APP_NAME}_selector", None)
    if value and value in app_names:
        st.session_state[ST_APP_NAME] = value


def render_refresh_button():
    if st.sidebar.button("↻ Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


def render_sidebar():
    apps = get_apps()
    app_name = st.session_state.get(ST_APP_NAME, None)

    if apps:
        app_names = sorted(
            list(set(app["app_name"] for app in apps)), reverse=True
        )

        if len(app_names) > 1:
            if not app_name or app_name not in app_names:
                app_idx = 0
            else:
                app_idx = app_names.index(app_name)
            if app_name := st.sidebar.selectbox(
                "Select an app",
                index=app_idx,
                key=f"{ST_APP_NAME}_selector",
                options=app_names,
                on_change=_handle_app_selection,
                args=(app_names,),
                disabled=len(app_names) == 1,
            ):
                st.query_params[ST_APP_NAME] = app_name

        else:
            app_name = app_names[0]

    render_refresh_button()

    with st.sidebar.expander("Info"):
        st.text(f"{core_package} {core_version}")
        st.text(f"{dashboard_package} {__version__}")

        st.link_button(
            "Share Feedback",
            "https://forms.gle/HAc4HBk5nZRpgw7C6",
            help="Help us improve TruLens!",
            use_container_width=True,
        )
    if app_name is None:
        st.error("No apps found in the database.")
    return app_name


def _factor_out_metadata(df: pd.DataFrame, metadata_col_name: str):
    metadata_cols = set()
    for _, row in df.iterrows():
        metadata_cols.update(row[metadata_col_name].keys())

    for metadata_key in metadata_cols:
        df[metadata_key] = df[metadata_col_name].apply(
            lambda x: x.get(metadata_key, None)
        )
    return df, metadata_cols


@st.cache_data(ttl=CACHE_TTL, show_spinner="Getting app versions")
def get_app_versions(app_name: str):
    app_versions = get_apps(app_name=app_name)
    app_versions_df = pd.DataFrame(app_versions)

    # Flatten metadata
    app_versions_df["metadata"] = app_versions_df["metadata"].apply(
        lambda x: flatten_metadata(x) if isinstance(x, dict) else {}
    )

    # Factor out metadata
    app_versions_df, app_version_metadata_cols = _factor_out_metadata(
        app_versions_df, "metadata"
    )

    app_versions_df = app_versions_df.replace({float("nan"): None})

    for bool_col in [PINNED_COL_NAME, EXTERNAL_APP_COL_NAME]:
        if bool_col in app_versions_df.columns:
            app_versions_df[bool_col] = (
                app_versions_df[bool_col] == "True"
            ).astype(bool)
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
    page_name_keys: Optional[List[str]] = None,
):
    for key in keys:
        if key == "filter.tags.multiselect":
            val = tags
        elif key.startswith("filter.metadata.") and key.endswith(
            ".multiselect"
        ):
            metadata_key = key[16:-12]
            val = metadata_options[metadata_key]
        else:
            val = ""

        st.session_state[key] = val
        query_param_key = key
        if page_name_keys and key in page_name_keys:
            query_param_key = ".".join(query_param_key.split(".")[1:])
        del st.query_params[query_param_key]


def render_app_version_filters(
    app_name: str,
    other_query_params_kv: Optional[dict[str, str]] = None,
    page_name_keys: Optional[List[str]] = None,
):
    app_versions_df, app_version_metadata_cols = get_app_versions(app_name)
    filtered_app_versions = app_versions_df

    col0, col1, col2 = st.columns(
        [0.7, 0.15, 0.15], vertical_alignment="bottom"
    )
    if other_query_params_kv:
        active_adv_filters = [k for k, v in other_query_params_kv.items() if v]
    else:
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
            args=(active_adv_filters, tags, metadata_options, page_name_keys),
        )

    return filtered_app_versions, app_version_metadata_cols
