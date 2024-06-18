
from streamlit_pills import pills
import pandas as pd
from trulens_eval.schema.feedback import FeedbackDefinition, FeedbackResult, FeedbackCall
import streamlit as st
from trulens_eval.utils.python import Future
from trulens_eval.ux.styles import CATEGORY
from trulens_eval.utils import display

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import threading

from pydantic import BaseModel

class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: list[FeedbackCall]
    icon: str

@st.experimental_fragment(run_every=2)
def select_feedback(record):
    feedback_cols = []
    feedbacks = {}
    icons = []
    for feedback, feedback_result in record.wait_for_feedback_results().items():
        print(feedback.name, feedback_result.result)
        call_data = {
                'feedback_definition': feedback,
                'feedback_name': feedback.name,
                'result': feedback_result.result
        }
        feedback_cols.append(call_data['feedback_name'])
        feedbacks[call_data['feedback_name']] = FeedbackDisplay(score=call_data['result'], calls=[], icon=get_icon(fdef=feedback, result=feedback_result.result))
        icons.append(feedbacks[call_data['feedback_name']].icon)
    
    st.write('**Feedback functions**')
    selected_feedback =  pills(
        "Feedback functions",
        feedback_cols,
        index=None,
        #format_func=lambda fcol: f"{fcol} {feedbacks[fcol].score:.4f}",
        #label_visibility="collapsed", # Hiding because we can't format the label here.
        icons=icons,
        key=f"{call_data['feedback_name']}_{len(feedbacks)}" # Important! Otherwise streamlit sometimes lazily skips update even with st.exprimental_fragment
    )
    return selected_feedback

@st.experimental_fragment(run_every=2)
def display_selected_feedback(record, selected_feedback):
    st.write("test")
    st.dataframe(display.get_feedback_result(record,
                                                 feedback_name = selected_feedback),
                                                 use_container_width=True,
                                                 hide_index=True
                                                 )

def get_icon(fdef: FeedbackDefinition, result: float):
    cat = CATEGORY.of_score(
        result or 0,
        higher_is_better=fdef.higher_is_better if fdef.higher_is_better is not None else True
    )
    return cat.icon

def update_result(fdef: FeedbackDefinition, fres: Future[FeedbackResult]):
    result = fres.result()
    calls = result.calls
    score = result.result or 0

    feedbacks={}
            
    feedbacks[fdef.name] = FeedbackDisplay(
                score=score, 
                calls=calls, 
                icon=get_icon(fdef, score)
            )

def st_thread(target, args) -> threading.Thread:
    """Return a function as a Streamlit-safe thread"""

    thread = threading.Thread(target=target, args=args)
    add_script_run_ctx(thread, get_script_run_ctx())
    return thread
