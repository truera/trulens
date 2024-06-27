import asyncio
import json
import math
from typing import List

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

from millify import millify
from pydantic import BaseModel
import streamlit as st
from streamlit_pills import pills

from trulens_eval import Tru
from trulens_eval.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema.feedback import FeedbackCall
from trulens_eval.schema.feedback import FeedbackDefinition
from trulens_eval.schema.record import Record
from trulens_eval.utils import display
from trulens_eval.utils.python import Future
from trulens_eval.ux import styles
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.styles import CATEGORY

class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: List[FeedbackCall]
    icon: str

def trulens_leaderboard(app_ids: List[str] = None):
    """
    Render the leaderboard page.

    Args:

        app_ids : A list of application IDs (default is None)

    !!! example

        ```python
        from trulens_eval import streamlit as trulens_st

        trulens_st.trulens_leaderboard()
        ```
    """
    tru = Tru()

    lms = tru.db
    df, feedback_col_names = lms.get_records_and_feedback([])
    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "") or
            row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }

    if df.empty:
        st.write("No records yet...")
        return

    df = df.sort_values(by="app_id", inplace=True)

    if df.empty:
        st.write("No records yet...")

    if app_ids is None:
    app_ids = list(df.app_id.unique())

    for app_id in app_ids:
        app_df = df.loc[df.app_id == app_id]
        if app_df.empty:
            continue
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        metadata = app_json.get("metadata")
        st.header(app, help=draw_metadata(metadata))
        app_feedback_col_names = [
            col_name for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols = st.columns(
            5 + len(app_feedback_col_names)
        )
        latency_mean = (
            app_df["latency"].
            apply(lambda td: td if td != MIGRATION_UNKNOWN_STR else None).mean()
        )

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            (
                f"{millify(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean) else "nan"
            ),
        )
        col3.metric(
            "Total Cost (USD)",
            f"${millify(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision = 2)}",
        )
        col4.metric(
            "Total Tokens",
            millify(
                sum(
                    tokens for tokens in app_df.total_tokens
                    if tokens is not None
                ),
                precision=2
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
                    delta_color="normal"
                )
            else:
                cat = CATEGORY.of_score(mean, higher_is_better=higher_is_better)
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal" if cat.compare(
                            mean, CATEGORY.PASS[cat.direction].threshold
                        ) else "inverse"
                    ),
                )

        st.markdown("""---""")

@st.experimental_fragment(run_every=2)
def trulens_feedback(record: Record):
    """
    Render clickable feedback pills for a given record.

    Args:

        record : A trulens record.

    !!! example

        ```python
        from trulens_eval import streamlit as trulens_st

        with tru_llm as recording:
            response = llm.invoke(input_text)

        record, response = recording.get()

        trulens_st.trulens_leaderboard()
        ```
    """
    feedback_cols = []
    feedbacks = {}
    icons = []
    for feedback, feedback_result in record.wait_for_feedback_results().items():
        call_data = {
            'feedback_definition': feedback,
            'feedback_name': feedback.name,
            'result': feedback_result.result
        }
        feedback_cols.append(call_data['feedback_name'])
        feedbacks[call_data['feedback_name']] = FeedbackDisplay(
            score=call_data['result'],
            calls=[],
            icon=_get_icon(fdef=feedback, result=feedback_result.result)
        )
        icons.append(feedbacks[call_data['feedback_name']].icon)

    st.write('**Feedback functions**')
    selected_feedback = pills(
        "Feedback functions",
        feedback_cols,
        index=None,
        format_func=lambda fcol: f"{fcol} {feedbacks[fcol].score:.4f}",
        label_visibility=
        "collapsed",  # Hiding because we can't format the label here.
        icons=icons,
        key=
        f"{call_data['feedback_name']}_{len(feedbacks)}"  # Important! Otherwise streamlit sometimes lazily skips update even with st.experimental_fragment
    )

    if selected_feedback is not None:
        st.dataframe(
            display.get_feedback_result(
                record, feedback_name=selected_feedback
            ),
            use_container_width=True,
            hide_index=True
        )


def _get_icon(fdef: FeedbackDefinition, result: float) -> str:
    """
    Get the icon for a given feedback definition and result.

    Args:

    fdef : FeedbackDefinition
        The feedback definition
    result : float
        The result of the feedback

    Returns:
        str: The icon for the feedback
    """
    cat = CATEGORY.of_score(
        result or 0,
        higher_is_better=fdef.higher_is_better
        if fdef.higher_is_better is not None else True
    )
    return cat.icon

def trulens_trace(record: Record):
    """
    Display the trace view for a record.

    Args:

        record : A trulens record.

    !!! example

        ```python
        from trulens_eval import streamlit as trulens_st

        with tru_llm as recording:
            response = llm.invoke(input_text)
            
        record, response = recording.get()

        trulens_st.trulens_leaderboard()
        ```
    """
    app_json = tru.get_app(app_id=record.app_id)
    record_json = _get_record_json(record)
    record_viewer(record_json=record_json, app_json=app_json)

def _get_record_json(record: Record) -> dict:
    """
    Get the JSON representation of a given record.

    Args:
        record: The record to get the JSON representation of

    Returns:
        dict: The JSON representation of the record
    """
    records, feedback = tru.get_records_and_feedback()
    record_json = records.loc[records['record_id'] == record.record_id
                             ]['record_json'].values[0]
    return json.loads(record_json)
