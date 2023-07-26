from st_aggrid import AgGrid
import streamlit as st
from ux.add_logo import add_logo

from trulens_eval import Tru
from trulens_eval.provider_apis import DEFAULT_RPM
from trulens_eval.schema import FeedbackResultStatus

st.set_page_config(page_title="Feedback Progress", layout="wide")

st.title("Feedback Progress")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

endpoints = ["OpenAI", "HuggingFace"]

tab1, tab2, tab3 = st.tabs(["Progress", "Endpoints", "Feedback Functions"])

with tab1:
    feedbacks = lms.get_feedback(
        status=[
            FeedbackResultStatus.NONE, FeedbackResultStatus.RUNNING,
            FeedbackResultStatus.FAILED
        ]
    )
    feedbacks = feedbacks.astype(str)
    data = AgGrid(
        feedbacks, allow_unsafe_jscode=True, fit_columns_on_grid_load=True
    )

with tab2:
    for e in endpoints:
        st.header(e)
        st.metric("RPM", DEFAULT_RPM)

with tab3:
    feedbacks = lms.get_feedback_defs()
    feedbacks = feedbacks.astype(str)
    data = AgGrid(
        feedbacks, allow_unsafe_jscode=True, fit_columns_on_grid_load=True
    )
