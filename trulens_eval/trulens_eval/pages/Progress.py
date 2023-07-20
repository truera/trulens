from datetime import datetime
import json
from typing import Dict, List

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from ux.add_logo import add_logo

from trulens_eval import db
from trulens_eval import Tru
from trulens_eval.db import DB
from trulens_eval.feedback import Feedback
from trulens_eval.provider_apis import Endpoint
from trulens_eval.provider_apis import HuggingfaceEndpoint
from trulens_eval.provider_apis import OpenAIEndpoint
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.util import is_empty
from trulens_eval.util import is_noserio
from trulens_eval.util import TP

from trulens_eval.keys import check_keys

check_keys(
    "OPENAI_API_KEY",
    "HUGGINGFACE_API_KEY"
)

st.set_page_config(page_title="Feedback Progress", layout="wide")

st.title("Feedback Progress")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

e_openai = OpenAIEndpoint()
e_hugs = HuggingfaceEndpoint()

endpoints = [e_openai, e_hugs]

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
        st.header(e.name.upper())
        st.metric("RPM", e.rpm)

with tab3:
    feedbacks = lms.get_feedback_defs()
    feedbacks = feedbacks.astype(str)
    data = AgGrid(
        feedbacks, allow_unsafe_jscode=True, fit_columns_on_grid_load=True
    )
