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
            details_json = json.loads(json.loads(details))
            chain_json = details_json['chain']
            if "llm" in chain_json:
                st.header("LLM Details:")
                llm_details_json = details_json["chain"]["llm"]
                llm_cols = st.columns(len(details_json["chain"]["llm"].items()))
                llm_keys = list(llm_details_json.keys())
                llm_values = list(llm_details_json.values())

                for i in range(len(llm_keys)):
                    with llm_cols[i]:
                        st.metric(llm_keys[i], llm_values[i])

            if "prompt" in chain_json:
                st.header("Prompt Type Details:")
                prompt_type_cols = st.columns(
                    len(details_json["chain"]["prompt"]["_type"].items())
                )
                prompt_types = details_json["chain"]["prompt"]["_type"]
                prompt_type_keys = list(prompt_types.keys())
                prompt_type_values = list(prompt_types.values())

                for i in range(len(prompt_type_keys)):
                    with prompt_type_cols[i]:
                        with st.expander(prompt_type_keys[i].capitalize(),
                                        expanded=True):
                            st.write(prompt_type_values[i])

                st.header("System Prompt Messages:")
                prompt_messages_json = details_json["chain"]["prompt"]["messages"][
                    0]["prompt"]
                prompt_messages_json = {
                    key: prompt_messages_json[key]
                    for key in prompt_messages_json
                    if key not in ["input_variables", "partial_variables"]
                }
                prompt_messages_cols = st.columns(len(prompt_messages_json.items()))
                prompt_messages_keys = list(prompt_messages_json.keys())
                prompt_messages_values = list(prompt_messages_json.values())

                for i in range(len(prompt_messages_keys)):
                    with prompt_messages_cols[i]:
                        with st.expander(prompt_messages_keys[i].capitalize(),
                                        expanded=True):
                            st.write(prompt_messages_values[i])

            if st.button("Display full json"):
                st.write(details_json)
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
