import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from trulens import core as mod_core
from trulens import dashboard as mod_dashboard
from trulens.core import experimental as core_experimental
from trulens.core import experimental as mod_experimental
from trulens.core import session as core_session
from trulens.core.database import base as core_db
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.utils import imports as import_utils
from trulens.dashboard import constants as dashboard_constants
from trulens.dashboard.components.record_viewer_otel import OtelSpan
from trulens.dashboard.components.record_viewer_otel import SpanRecord
from trulens.dashboard.components.record_viewer_otel import SpanTrace
from trulens.dashboard.utils import metadata_utils
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.otel.semconv.trace import ResourceAttributes

ST_APP_NAME = "app_name"
ST_RECORDS_LIMIT = "records_limit"


def set_page_config(page_title: Optional[str] = None):
    page_title = f"TruLens: {page_title}" if page_title else "TruLens"
    st.set_page_config(
        page_title=page_title,
        page_icon="https://www.trulens.org/img/favicon.ico",
        layout="wide",
    )

    if is_sis_compatibility_enabled():
        pass
    else:
        logo = str(
            import_utils.static_resource("dashboard", "ux/trulens_logo.svg")
        )
        logo_small = str(
            import_utils.static_resource("dashboard", "ux/trulens_squid.svg")
        )
        st.logo(logo, icon_image=logo_small, link="https://www.trulens.org/")

    if ST_RECORDS_LIMIT not in st.session_state:
        st.session_state[ST_RECORDS_LIMIT] = dashboard_constants.RECORDS_LIMIT


def add_query_param(param_name: str, param_value: str):
    st.query_params[param_name] = param_value


def read_query_params_into_session_state(
    page_name: str, transforms: Optional[Dict[str, Callable[[str], Any]]] = None
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


def is_sis_compatibility_enabled():
    """This method returns whether the SIS compatibility feature is enabled.
    The SiS compatibility feature adapts dashboard components to support Streamlit in Snowflake (SiS).
    As of 11/13/2024, SiS runs on Python 3.8, Streamlit 1.35.0, and does not support bidirectional custom components.

    In the TruLens dashboard, this flag will replace or disable certain custom components (like Aggrid and the trace viewer).

    Returns:
        bool: True if the SIS compatibility feature is enabled, False otherwise.
    """
    return get_session().experimental_feature(
        core_experimental.Feature.SIS_COMPATIBILITY
    )


@st.cache_resource(show_spinner="Setting up TruLens session")
def get_session() -> core_session.TruSession:
    """Parse command line arguments and initialize TruSession with them.

    As TruSession is a singleton, further TruSession() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--snowflake-account", default=None)
    parser.add_argument("--snowflake-user", default=None)
    parser.add_argument("--snowflake-role", default=None)
    parser.add_argument("--snowflake-database", default=None)
    parser.add_argument("--snowflake-schema", default=None)
    parser.add_argument("--snowflake-warehouse", default=None)
    parser.add_argument("--snowflake-authenticator", default=None)
    parser.add_argument(
        "--snowflake-use-account-event-table", action="store_true"
    )
    parser.add_argument("--sis-compatibility", action="store_true")
    parser.add_argument(
        "--database-prefix", default=core_db.DEFAULT_DATABASE_PREFIX
    )
    parser.add_argument(
        "--otel-tracing",
        action="store_true",
        help="Enable OTEL tracing in the dashboard",
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    if args.snowflake_account:
        from snowflake.snowpark import Session
        from trulens.connectors.snowflake import SnowflakeConnector

        connection_params = {
            "account": args.snowflake_account,
            "user": args.snowflake_user,
            "role": args.snowflake_role,
            "database": args.snowflake_database,
            "schema": args.snowflake_schema,
            "warehouse": args.snowflake_warehouse,
            "authenticator": args.snowflake_authenticator,
        }
        use_account_event_table = bool(args.snowflake_use_account_event_table)
        snowpark_session = Session.builder.configs(connection_params).create()
        session = core_session.TruSession(
            connector=SnowflakeConnector(
                snowpark_session=snowpark_session,
                use_account_event_table=use_account_event_table,
                database_prefix=args.database_prefix,
            )
        )
    else:
        session = core_session.TruSession(
            database_url=args.database_url, database_prefix=args.database_prefix
        )

    if args.sis_compatibility:
        session.experimental_enable_feature(
            mod_experimental.Feature.SIS_COMPATIBILITY
        )

    # Store the otel_tracing flag in the session state
    if args.otel_tracing:
        os.environ["TRULENS_OTEL_TRACING"] = "1"

    return session


@st.cache_data(
    ttl=dashboard_constants.CACHE_TTL, show_spinner="Getting record data"
)
def get_records_and_feedback(
    app_ids: Optional[List[str]] = None,
    app_name: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
):
    session = get_session()
    lms = session.connector.db
    assert lms

    records_df, feedback_col_names = lms.get_records_and_feedback(
        app_ids=app_ids,
        app_name=app_name,
        offset=offset,
        limit=limit,
    )

    records_df["record_metadata"] = records_df["record_json"].apply(
        lambda x: metadata_utils.flatten_metadata(x["meta"])
        if isinstance(x["meta"], dict)
        else {}
    )

    records_df, _ = _factor_out_metadata(records_df, "record_metadata")

    if dashboard_constants.HIDE_RECORD_COL_NAME in records_df.columns:
        records_df[dashboard_constants.HIDE_RECORD_COL_NAME] = (
            records_df[dashboard_constants.HIDE_RECORD_COL_NAME] == "True"
        ).astype(bool)
    records_df = records_df.replace({float("nan"): None})

    feedback_col_names = [
        col for col in feedback_col_names if col in records_df.columns
    ]

    return records_df, feedback_col_names


@st.cache_data(
    ttl=dashboard_constants.CACHE_TTL, show_spinner="Getting app data"
)
def get_apps(app_name: Optional[str] = None):
    session = get_session()
    lms = session.connector.db
    assert lms
    return list(lms.get_apps(app_name=app_name))


@st.cache_data(
    ttl=dashboard_constants.CACHE_TTL,
    show_spinner="Getting feedback definitions",
)
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
        st.query_params.clear()
        st.session_state.clear()
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
        st.text(f"{mod_core.__package__} {mod_core.__version__}")
        st.text(f"{mod_dashboard.__package__} {mod_dashboard.__version__}")

        BUG_REPORT_URL = "https://github.com/truera/trulens/issues/new?template=bug-report.md"

        st.link_button(
            "Report a Bug 🐞",
            BUG_REPORT_URL,
            help="Help us fix bugs! (Emoji: Ladybug)",
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


@st.cache_data(
    ttl=dashboard_constants.CACHE_TTL, show_spinner="Getting app versions"
)
def get_app_versions(app_name: str):
    app_versions = get_apps(app_name=app_name)
    app_versions_df = pd.DataFrame(app_versions)

    # Flatten metadata
    app_versions_df["metadata"] = app_versions_df["metadata"].apply(
        lambda x: metadata_utils.flatten_metadata(x)
        if isinstance(x, dict)
        else {}
    )

    # Factor out metadata
    app_versions_df, app_version_metadata_cols = _factor_out_metadata(
        app_versions_df, "metadata"
    )

    app_versions_df = app_versions_df.replace({float("nan"): None})

    for bool_col in [
        dashboard_constants.PINNED_COL_NAME,
        dashboard_constants.EXTERNAL_APP_COL_NAME,
    ]:
        if bool_col in app_versions_df.columns:
            app_versions_df[bool_col] = (
                app_versions_df[bool_col] == "True"
            ).astype(bool)
    return app_versions_df, list(app_version_metadata_cols)


def _get_query_args_handler(key: str, max_options: Optional[int] = None):
    new_val = st.session_state.get(key, None)
    if not new_val:
        # if no new value, remove query param
        if key in st.query_params:
            del st.query_params[key]
        return
    elif isinstance(new_val, list):
        if len(new_val) == max_options:
            # don't need to explicitly add query args as default is all options
            if key in st.query_params:
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
        if query_param_key in st.query_params:
            del st.query_params[query_param_key]


def render_app_version_filters(
    app_name: str,
    other_query_params_kv: Optional[Dict[str, str]] = None,
    page_name_keys: Optional[List[str]] = None,
):
    app_versions_df, app_version_metadata_cols = get_app_versions(app_name)
    filtered_app_versions = app_versions_df

    col0, col1, col2 = st_columns(
        [0.7, 0.15, 0.15], vertical_alignment="bottom"
    )
    if other_query_params_kv:
        active_adv_filters = [k for k, v in other_query_params_kv.items() if v]
    else:
        active_adv_filters = []

    st.session_state.setdefault("filter.search", "")
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

        if len(selected_tags):
            filtered_app_versions = filtered_app_versions.loc[
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


def _parse_json_fields(field: Any) -> Dict[str, Any]:
    """Parse a JSON field from the database, handling potential errors.

    Args:
        field: The field to parse, can be a string or dict

    Returns:
        Parsed dictionary or error dictionary if parsing fails
    """
    if isinstance(field, dict):
        return field
    if isinstance(field, str):
        try:
            return json.loads(field)
        except Exception as e:
            return {"error": f"Unable to parse {field}: {e}"}
    return {"error": f"Invalid {field} format"}


def _convert_timestamp(ts: Any) -> Union[int, float]:
    """Convert various timestamp formats to Unix timestamp in seconds.

    Args:
        ts: Timestamp in any supported format (int, float, str, pd.Timestamp)

    Returns:
        Unix timestamp in seconds, or 0 if conversion fails
    """
    if pd.isna(ts):
        return 0
    elif isinstance(ts, (int, float)):
        return int(ts)
    elif isinstance(ts, str):
        return int(pd.Timestamp(ts).timestamp())
    elif isinstance(ts, pd.Timestamp):
        return int(ts.timestamp())
    else:
        return 0


def _make_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable format.

    Args:
        value: Any value to convert

    Returns:
        JSON-serializable version of the value
    """
    try:
        json.dumps({"test": value})
        return value
    except (TypeError, OverflowError):
        return str(value)


def _map_event_to_otel_span(row: pd.Series) -> Optional[OtelSpan]:
    """Convert an Event ORM table row to an OtelSpan format.

    Args:
        row: A pandas Series containing the Event ORM table row data

    Returns:
        An OtelSpan object if conversion is successful, None otherwise
    """
    try:
        # Parse record data
        record_data = _parse_json_fields(row.get("record", {}))
        span_record: SpanRecord = {
            "name": str(record_data.get("name", "")),
            "parent_span_id": str(record_data.get("parent_span_id", "")),
            "status": str(record_data.get("status", "")),
        }

        # Parse trace data
        trace_data = _parse_json_fields(row.get("trace", {}))
        span_trace: SpanTrace = {
            "trace_id": str(trace_data.get("trace_id", "")),
            "parent_id": str(trace_data.get("parent_id", "")),
            "span_id": str(trace_data.get("span_id", "")),
        }

        # Process record attributes
        record_attributes = _parse_json_fields(row.get("record_attributes", {}))
        serializable_attributes = {
            k: _make_serializable(v) for k, v in record_attributes.items()
        }

        # Create span with converted timestamps
        span: OtelSpan = {
            "event_id": str(row.get("event_id", "")),
            "record": span_record,
            "record_attributes": serializable_attributes,
            "start_timestamp": _convert_timestamp(row.get("start_timestamp")),
            "timestamp": _convert_timestamp(row.get("timestamp")),
            "trace": span_trace,
        }

        return span
    except Exception as e:
        st.warning(f"Error processing span: {e}")
        return None


def _convert_events_to_otel_spans(events_df: pd.DataFrame) -> List[OtelSpan]:
    """Convert a DataFrame of Event ORM table rows to a list of OtelSpans.

    Args:
        events_df: DataFrame containing Event ORM table rows

    Returns:
        A list of OtelSpans for all successfully converted events
    """
    serializable_spans: List[OtelSpan] = []

    for _, row in events_df.iterrows():
        if span := _map_event_to_otel_span(row):
            serializable_spans.append(span)

    return serializable_spans


@st.cache_data(
    ttl=dashboard_constants.CACHE_TTL, show_spinner="Getting events for record"
)
def _get_event_otel_spans(record_id: str) -> List[OtelSpan]:
    """Get all event spans for a given record ID.

    Args:
        record_id: The record ID to get events for.

    Returns:
        A list of OtelSpans for all events corresponding to the given record ID.
    """
    session = get_session()
    db = session.connector.db

    if not db or not hasattr(db, "get_events_by_record_id"):
        st.error(
            f"Error getting events by record {record_id}: database must support OTEL spans"
        )
        return []

    try:
        events_df = db.get_events_by_record_id(record_id)
        return _convert_events_to_otel_spans(events_df)
    except Exception as e:
        st.error(f"Error getting events for record {record_id}: {e}")
        return []


def _check_cross_format_records(
    app_name: Optional[str] = None,
    app_ids: Optional[List[str]] = None,
) -> tuple[int, int]:
    """Check record counts in both OTEL and non-OTEL formats.

    Returns:
        Tuple of (otel_count, non_otel_count)
    """
    session = get_session()
    otel_count = 0
    non_otel_count = 0

    if isinstance(session.connector.db, SQLAlchemyDB):
        db = session.connector.db  # type: ignore

        with db.session.begin() as session_ctx:  # type: ignore
            # Check OTEL records (EVENT table)
            query = sa.select(sa.func.count(db.orm.Event.event_id))  # type: ignore

            if app_name:
                # For OTEL events, app_name is in resource_attributes JSON
                app_name_expr = db._json_extract_otel(
                    "resource_attributes", ResourceAttributes.APP_NAME
                )
                query = query.where(app_name_expr == app_name)
            elif app_ids:
                # For OTEL events, app_id is in resource_attributes JSON
                app_id_expr = db._json_extract_otel(
                    "resource_attributes", ResourceAttributes.APP_ID
                )
                query = query.where(app_id_expr.in_(app_ids))

            result = session_ctx.execute(query).scalar()
            otel_count = result or 0

            # Check non-OTEL records (RECORD table)
            query = sa.select(sa.func.count(db.orm.Record.record_id))  # type: ignore

            if app_name:
                # For non-OTEL records, need to join with AppDefinition
                query = query.join(db.orm.Record.app).where(  # type: ignore
                    db.orm.AppDefinition.app_name == app_name  # type: ignore
                )
            elif app_ids:
                query = query.where(db.orm.Record.app_id.in_(app_ids))  # type: ignore

            result = session_ctx.execute(query).scalar()
            non_otel_count = result or 0

    return otel_count, non_otel_count


def _show_no_records_error(
    app_name: Optional[str] = None, app_ids: Optional[List[str]] = None
) -> None:
    """Show helpful error message when no records found, with cross-format record counts."""
    is_otel_mode = is_otel_tracing_enabled()
    otel_count, non_otel_count = _check_cross_format_records(app_name, app_ids)

    if is_otel_mode and otel_count == 0 and non_otel_count > 0:
        st.error(
            f"No records found for app `{app_name}` in OTEL mode. "
            f"However, {non_otel_count} records exist in non-OTEL format. "
            f"Restart without `TRULENS_OTEL_TRACING` to access them.",
            icon="🔄",
        )
    elif not is_otel_mode and non_otel_count == 0 and otel_count > 0:
        st.error(
            f"No records found for app `{app_name}` in non-OTEL mode. "
            f"However, {otel_count} records exist in OTEL format. "
            f"Set `TRULENS_OTEL_TRACING=1` to access them.",
            icon="🔄",
        )
    else:
        st.error(f"No records found for app `{app_name}`.")
