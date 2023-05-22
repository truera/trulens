from datetime import datetime
import json
from typing import Dict, List

from trulens_eval.keys import *

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP
from ux.add_logo import add_logo
from trulens_eval.tru_db import is_empty


from trulens_eval import tru_db
from trulens_eval.provider_apis import Endpoint

from trulens_eval.tru_db import is_noserio
from trulens_eval.tru_db import TruDB

st.set_page_config(page_title="Computations", layout="wide")

st.title("Computations")

st.runtime.legacy_caching.clear_cache()

add_logo()

lms = tru_db.LocalSQLite()

e_openai = Endpoint("openai")
e_hugs = Endpoint("huggingface")
e_cohere = Endpoint("cohere")

endpoints = [e_openai, e_hugs, e_cohere]

tab1, tab2, tab3 = st.tabs(["Queue", "Endpoints", "Feedback Functions"])

def prepare_feedback(row):
    record_json = row.record_json

    st.write(row.record_json)
    st.write(row.feedback_json)

    feedback = Feedback.of_json(row.feedback_json)
    feedback.run_and_log(record_json=record_json, db=lms)

with tab1:
    # st.write(TP().status())

    st.metric("Running computations (probably wrong)", TP().running)

    feedbacks = lms.get_feedback()
    st.write(feedbacks)

    for i, row in feedbacks.iterrows():
        if row.status == 0:
            st.write(f"Starting run for row {i}.")
            TP().runlater(prepare_feedback, row)
        elif row.status in [-1, 1]:
            now = datetime.now().timestamp()
            if now - row.last_ts > 30:
                st.write(f"Incomplete row {i} last made progress over 30 seconds ago. Retrying.")
                TP().runlater(prepare_feedback, row)
            else:
                st.write(f"Incomplete row {i} last made progress less than 30 seconds ago. Giving it more time.")

        elif row.status == 2:
            pass
            #st.write(f"Row {i} already done.")

with tab2:
    for e in endpoints:
        st.header(e.name.upper())
        st.metric("RPM", e.rpm)
        st.write(e.tqdm)

with tab3:
    feedbacks = lms.get_feedback_defs()
    st.write(feedbacks)
