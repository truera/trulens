import ast
import json

import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import streamlit as st

import tru_db
from ux.add_logo import add_logo

st.runtime.legacy_caching.clear_cache()

add_logo()

lms = tru_db.LocalModelStore()
df, df_feedback = lms.get_records_and_feedback([])

if 'Chains' in st.session_state:
    model = st.session_state.chain
else:
    model = None

models = list(df.chain_id.unique())

options = st.multiselect('Choose a model', models, default=model)

col0, col1, col2, col3 = st.columns(4)
model_df = df.loc[df.chain_id.isin(options)]
model_df_feedback = df_feedback.loc[df.chain_id.isin(options)]
col0.metric("Name", model)
col1.metric("Records", len(model_df))
col2.metric("Cost", sum(df.total_cost))
col3.metric("Feedback Results", len(df_feedback.columns))

if (len(options) == 0):
    st.header("All Chains")
else:
    st.header(options)

tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

#mds = model_store.ModelDataStore()

with tab1:
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="single", use_checkbox=False)

    #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    data = AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED
    )

    selected_rows = data['selected_rows']
    selected_rows = pd.DataFrame(selected_rows)

    if len(selected_rows) != 0:
        details = selected_rows['details'][0]
        #print(details)
        st.json(json.loads(details))

with tab2:
    feedback = df_feedback.columns
    cols = 4
    rows = len(feedback) // cols + 1

    for row_num in range(rows):
        with st.container():
            columns = st.columns(cols)
            for col_num in range(cols):
                with columns[col_num]:
                    ind = row_num * cols + col_num
                    if ind < len(feedback):
                        st.text(feedback[ind])
                        print(model_df_feedback[feedback[ind]])
                        st.bar_chart(model_df_feedback[feedback[ind]])
