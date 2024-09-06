import argparse
import asyncio
import json
import sys
from typing import List

from pydantic import BaseModel
import streamlit as st
from streamlit_pills import pills
from trulens.core import TruSession
from trulens.core.database.base import DEFAULT_DATABASE_PREFIX
from trulens.core.schema.feedback import FeedbackCall
from trulens.core.schema.record import Record
from trulens.core.utils.json import json_str_of_obj
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.display import get_feedback_result
from trulens.dashboard.display import get_icon

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
        st.dataframe(
            get_feedback_result(record, feedback_name=selected_feedback),
            use_container_width=True,
            hide_index=True,
        )


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
