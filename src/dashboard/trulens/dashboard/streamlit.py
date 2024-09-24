import argparse
import asyncio
import json
import math
import sys
from typing import List

from pydantic import BaseModel
import streamlit as st
from streamlit_pills import pills
from trulens.core import TruSession
from trulens.core.database.base import DEFAULT_DATABASE_PREFIX
from trulens.core.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens.core.schema.feedback import FeedbackCall
from trulens.core.schema.record import Record
from trulens.core.utils.json import json_str_of_obj
from trulens.core.utils.text import format_quantity
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.display import expand_groundedness_df
from trulens.dashboard.display import get_feedback_result
from trulens.dashboard.display import get_icon
from trulens.dashboard.display import highlight
from trulens.dashboard.ux import styles
from trulens.dashboard.ux.components import draw_metadata_and_tags

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())


class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: List[FeedbackCall]
    icon: str


def init_from_args():
    """Parse command line arguments and initialize Tru with them.

    As Tru is a singleton, further TruSession() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--database-prefix", default=DEFAULT_DATABASE_PREFIX)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    TruSession(
        database_url=args.database_url, database_prefix=args.database_prefix
    )


def trulens_leaderboard(app_ids: List[str] = None):
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
    session = TruSession()

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
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        app_name = app_json["app_name"]
        app_version = app_json["app_version"]
        app_name_version = f"{app_name} - {app_version}"
        metadata = app_json.get("metadata")
        tags = app_json.get("tags")
        st.header(app_name_version, help=draw_metadata_and_tags(metadata, tags))
        app_feedback_col_names = [
            col_name
            for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols = st.columns(
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
                cat = styles.CATEGORY.of_score(
                    mean, higher_is_better=higher_is_better, is_distance=True
                )
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal",
                )
            else:
                cat = styles.CATEGORY.of_score(
                    mean, higher_is_better=higher_is_better
                )
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal"
                        if cat.compare(
                            mean, styles.CATEGORY.PASS[cat.direction].threshold
                        )
                        else "inverse"
                    ),
                )

        st.markdown("""---""")


@st.fragment(run_every=2)
def trulens_feedback(record: Record):
    """
    Render clickable feedback pills for a given record.

    Args:

        record (Record): A trulens record.

    Example:
        ```python
        from trulens.core import streamlit as trulens_st

        with tru_llm as recording:
            response = llm.invoke(input_text)

        record, response = recording.get()

        trulens_st.trulens_feedback(record=record)
        ```
    """
    feedback_cols = []
    feedbacks = {}
    icons = []
    default_direction = "HIGHER_IS_BETTER"
    session = TruSession()
    lms = session.connector.db
    feedback_defs = lms.get_feedback_defs()

    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "")
            or row.feedback_json["implementation"]["name"]
        ): (
            "HIGHER_IS_BETTER"
            if row.feedback_json.get("higher_is_better", True)
            else "LOWER_IS_BETTER"
        )
        for _, row in feedback_defs.iterrows()
    }

    for feedback, feedback_result in record.wait_for_feedback_results().items():
        call_data = {
            "feedback_definition": feedback,
            "feedback_name": feedback.name,
            "result": feedback_result.result,
        }
        feedback_cols.append(call_data["feedback_name"])
        feedbacks[call_data["feedback_name"]] = FeedbackDisplay(
            score=call_data["result"],
            calls=[],
            icon=get_icon(fdef=feedback, result=feedback_result.result),
        )
        icons.append(feedbacks[call_data["feedback_name"]].icon)

    st.header("Feedback Functions")
    selected_feedback = pills(
        "Feedback functions",
        feedback_cols,
        index=None,
        format_func=lambda fcol: f"{fcol} {feedbacks[fcol].score:.4f}",
        label_visibility="collapsed",  # Hiding because we can't format the label here.
        icons=icons,
        key=f"{call_data['feedback_name']}_{len(feedbacks)}",  # Important! Otherwise streamlit sometimes lazily skips update even with st.fragment
    )

    if selected_feedback is not None:
        df = get_feedback_result(record, feedback_name=selected_feedback)
        if "groundedness" in selected_feedback.lower():
            df = expand_groundedness_df(df)
        else:
            pass

        # Apply the highlight function row-wise
        styled_df = df.style.apply(
            lambda row: highlight(
                row,
                selected_feedback=selected_feedback,
                feedback_directions=feedback_directions,
                default_direction=default_direction,
            ),
            axis=1,
        )

        # Format only numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            styled_df = styled_df.format({col: "{:.2f}"})

        st.dataframe(styled_df, hide_index=True)


def trulens_trace(record: Record):
    """
    Display the trace view for a record.

    Args:

        record (Record): A trulens record.

    Example:
        ```python
        from trulens.core import streamlit as trulens_st

        with tru_llm as recording:
            response = llm.invoke(input_text)

        record, response = recording.get()

        trulens_st.trulens_trace(record=record)
        ```
    """

    session = TruSession()
    app = session.get_app(app_id=record.app_id)
    record_viewer(record_json=json.loads(json_str_of_obj(record)), app_json=app)
