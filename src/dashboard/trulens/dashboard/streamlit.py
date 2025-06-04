import argparse
import asyncio
import json
import math
import sys
from typing import List, Optional, Union

from pydantic import BaseModel
import streamlit as st
from trulens.core import session as core_session
from trulens.core.database import base as core_db
from trulens.core.database.legacy import migration as legacy_migration
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import text as text_utils
from trulens.dashboard.components import (
    record_viewer as dashboard_record_viewer,
)
from trulens.dashboard.components import (
    record_viewer_otel as dashboard_record_viewer_otel,
)
from trulens.dashboard.utils import dashboard_utils
from trulens.dashboard.utils import streamlit_compat
from trulens.dashboard.utils.dashboard_utils import _get_event_otel_spans
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux import components as dashboard_components
from trulens.dashboard.ux import styles as dashboard_styles

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())


class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: List[feedback_schema.FeedbackCall]
    icon: str


def init_from_args():
    """Parse command line arguments and initialize Tru with them.

    As Tru is a singleton, further TruSession() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--database-prefix", default=core_db.DEFAULT_DATABASE_PREFIX
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    core_session.TruSession(
        database_url=args.database_url, database_prefix=args.database_prefix
    )


def trulens_leaderboard(app_ids: Optional[List[str]] = None):
    """
    Render the leaderboard page.

    Args:

        app_ids List[str]: A list of application IDs (default is None)

    Example:
        ```python
        from trulens.core import streamlit as trulens_st

        trulens_st.trulens_leaderboard()
        ```
    """
    session = core_session.TruSession()

    lms = session.connector.db
    df, feedback_col_names = lms.get_records_and_feedback(app_ids=app_ids)
    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "")
            or row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }

    if df.empty:
        st.write("No records yet...")
        return

    df.sort_values(by="app_id", inplace=True)

    if df.empty:
        st.write("No records yet...")

    def get_data():
        return lms.get_records_and_feedback([])

    def get_apps():
        return list(lms.get_apps())

    records, feedback_col_names = get_data()
    records = records.sort_values(by="app_id")

    apps = get_apps()

    for app in apps:
        app_df = records.loc[records.app_id == app]
        if app_df.empty:
            continue
        app_json = app_df["app_json"].iloc[0]
        app_name = app_json["app_name"]
        app_version = app_json["app_version"]
        app_name_version = f"{app_name} - {app_version}"
        metadata = app_json.get("metadata")
        tags = app_json.get("tags")
        st.header(
            app_name_version,
            help=dashboard_components.draw_metadata_and_tags(metadata, tags),
        )
        app_feedback_col_names = [
            col_name
            for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols = st_columns(
            5 + len(app_feedback_col_names)
        )
        latency_mean = (
            app_df["latency"]
            .apply(
                lambda td: td
                if td != legacy_migration.MIGRATION_UNKNOWN_STR
                else None
            )
            .mean()
        )

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            (
                f"{text_utils.format_quantity(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean)
                else "nan"
            ),
        )
        col3.metric(
            "Total Cost (USD)",
            f"${text_utils.format_quantity(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision=2)}",
        )
        col4.metric(
            "Total Tokens",
            text_utils.format_quantity(
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
                dashboard_styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            higher_is_better = feedback_directions.get(col_name, True)

            if "distance" in col_name:
                cat = dashboard_styles.CATEGORY.of_score(
                    mean, higher_is_better=higher_is_better, is_distance=True
                )
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal",
                )
            else:
                cat = dashboard_styles.CATEGORY.of_score(
                    mean, higher_is_better=higher_is_better
                )
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal"
                        if cat.compare(
                            mean,
                            dashboard_styles.CATEGORY.PASS[
                                cat.direction
                            ].threshold,
                        )
                        else "inverse"
                    ),
                )

        st.markdown("""---""")


@streamlit_compat.st_fragment(run_every=2)
def trulens_feedback(record: Union[record_schema.Record, str]):
    """Render clickable feedback pills for a given record.

    Args:
        record: Either a trulens record (non-OTel) or a record_id string (OTel).

    Example:
        ```python
        from trulens.core import streamlit as trulens_st

        with tru_llm as recording:
            response = llm.invoke(input_text)

        record, response = recording.get()

        trulens_st.trulens_feedback(record=record)
        ```
    """
    session = core_session.TruSession()
    lms = session.connector.db

    _, feedback_directions = dashboard_utils.get_feedback_defs()

    if isinstance(record, record_schema.Record):
        record_id = record.record_id
    elif isinstance(record, str):
        record_id = record

    records_df, feedback_col_names = lms.get_records_and_feedback()
    # TODO: filter by record id
    selected_record_row = records_df[records_df["record_id"] == record_id]

    if not selected_record_row.empty:
        selected_record_row = selected_record_row.iloc[0]
    else:
        st.warning(f"No record found with record_id: {record_id}")
        return

    if selected_ff := _render_feedback_pills(
        feedback_col_names=feedback_col_names,
        selected_row=selected_record_row,
        feedback_directions=feedback_directions,
    ):
        _render_feedback_call(
            selected_ff,
            selected_record_row,
            feedback_directions=feedback_directions,
        )


def trulens_trace(record: Union[record_schema.Record, str]):
    """Display the trace view for a record.

    Args:
        record: Either a trulens record (non-OTel) or a record_id string (OTel).

    Example:
        ```python
        from trulens.core import streamlit as trulens_st

        # Using with Record object
        with tru_llm as recording:
            response = llm.invoke(input_text)
        record, response = recording.get()
        trulens_st.trulens_trace(record=record)

        # Using with record_id string
        trulens_st.trulens_trace(record="record_123")
        ```
    """

    session = core_session.TruSession()
    if isinstance(record, record_schema.Record):
        app = session.get_app(app_id=record.app_id)
    if dashboard_utils.is_sis_compatibility_enabled():
        st.warning(
            "TruLens trace view is not enabled when SiS compatibility is enabled."
        )
    elif isinstance(record, str):
        event_spans = _get_event_otel_spans(record)
        if event_spans:
            dashboard_record_viewer_otel.record_viewer_otel(
                spans=event_spans, key=None
            )
        else:
            st.warning("No OTel trace data available for this record.")
    else:
        dashboard_record_viewer.record_viewer(
            record_json=json.loads(json_utils.json_str_of_obj(record)),
            app_json=app,
        )
