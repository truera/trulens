import json

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from ux.add_logo import add_logo

from trulens_evalchain import tru_db

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo()

lms = tru_db.LocalSQLite()
df, df_feedback = lms.get_records_and_feedback([])

if df.empty:
    st.write("No records yet...")
else:
    chains = list(df.chain_id.unique())

    if 'Chains' in st.session_state:
        chain = st.session_state.chain
    else:
        chain = chains

    options = st.multiselect('Filter Chains', chains, default=chain)

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
        evaluations_df = df.copy()
        evaluations_df = evaluations_df.merge(
            df_feedback, left_index=True, right_index=True
        )
        evaluations_df = evaluations_df[[
            'record_id',
            'chain_id',
            'input',
            'output',
        ] + list(df_feedback.columns) + ['tags', 'ts', 'details']]
        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        cellstyle_jscode = JsCode(
            """
        function(params) {
            if (parseFloat(params.value) < 0.5) {
                return {
                    'color': 'black',
                    'backgroundColor': '#FCE6E6'
                }
            } else if (parseFloat(params.value) >= 0.5) {
                return {
                    'color': 'black',
                    'backgroundColor': '#4CAF50'
                }
            } else {
                return {
                    'color': 'black',
                    'backgroundColor': 'white'
                }
            }
        };
        """
        )

        gb.configure_column(
            'record_id', header_name='Record ID', maxWidth=100, pinned='left'
        )
        gb.configure_column(
            'chain_id', header_name='Chain ID', maxWidth=100, pinned='left'
        )
        gb.configure_column(
            'input', header_name='User Input'
            #, minWidth=500
        )
        gb.configure_column(
            'output',
            header_name='Response',
            #minWidth=500
        )
        gb.configure_column(
            'tags', header_name='Tags', minWidth=100, pinned='right'
        )
        gb.configure_column(
            'ts', header_name='Time Stamp', minWidth=100, pinned='right'
        )
        # gb.configure_column('details', maxWidth=0)

        for feedback_col in df_feedback.columns:
            gb.configure_column(feedback_col, cellStyle=cellstyle_jscode)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)

        #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True
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
