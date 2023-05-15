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

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo()

lms = tru_db.LocaSQLite()
df, df_feedback = lms.get_records_and_feedback([])

if df.empty:
    st.write("No records yet...")
else:

    if 'Chains' in st.session_state:
        model = st.session_state.chain
    else:
        model = None

    models = list(df.chain_id.unique())

    options = st.multiselect('Filter Chains', models, default=model)

    model_df = df.loc[df.chain_id.isin(options)]
    model_df_feedback = df_feedback.loc[df.chain_id.isin(options)]

    if (len(options) == 0):
        st.header("All Chains")
    elif (len(options) == 1):
        st.header(options[0])
    else:
        st.header("Multiple Chains Selected")

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    #mds = model_store.ModelDataStore()

    with tab1:
        evaluations_df = df.drop('feedback', axis=1)
        evaluations_df = evaluations_df.merge(
            df_feedback, left_index=True, right_index=True
        )
        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)

        #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED
        )

        selected_rows = data['selected_rows']
        selected_rows = pd.DataFrame(selected_rows)
        st.write("Hint: select a row to display chain metadata")

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
