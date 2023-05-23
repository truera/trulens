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


from trulens_eval import tru_db, Tru
from trulens_eval.provider_apis import Endpoint

from trulens_eval.tru_db import is_noserio
from trulens_eval.tru_db import TruDB

st.set_page_config(page_title="Feedback Progress", layout="wide")

st.title("Feedback Progress")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

e_openai = Endpoint("openai")
e_hugs = Endpoint("huggingface")
e_cohere = Endpoint("cohere")

endpoints = [e_openai, e_hugs, e_cohere]

tab1, tab2, tab3 = st.tabs(["Progress", "Endpoints", "Feedback Functions"])

with tab1:
    feedbacks = lms.get_feedback(status=[-1,0,1])
    st.write(feedbacks)

with tab2:
    for e in endpoints:
        st.header(e.name.upper())
        st.metric("RPM", e.rpm)
        st.write(e.tqdm)

with tab3:
    feedbacks = lms.get_feedback_defs()
    st.write(feedbacks)
