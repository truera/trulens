import json
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from ux.add_logo import add_logo

from trulens_eval import Tru
from trulens_eval import tru_db
from trulens_eval.tru_db import is_empty
from trulens_eval.tru_db import is_noserio
from trulens_eval.tru_db import TruDB

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

df_results, feedback_cols = lms.get_records_and_feedback([])

if df_results.empty:
    st.write("No records yet...")

else:
    chains = list(df_results.chain_id.unique())

    if 'Chains' in st.session_state:
        chain = st.session_state.chain
    else:
        chain = chains

    options = st.multiselect('Filter Chains', chains, default=chain)

    if (len(options) == 0):
        st.header("All Chains")
        chain_df = df_results

    elif (len(options) == 1):
        st.header(options[0])

        chain_df = df_results[df_results.chain_id.isin(options)]

    else:
        st.header("Multiple Chains Selected")

        chain_df = df_results[df_results.chain_id.isin(options)]

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:
        gridOptions = {'alwaysShowHorizontalScroll': True}
        evaluations_df = chain_df
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

        gb.configure_column('record_id', header_name='Record ID')
        gb.configure_column('chain_id', header_name='Chain ID')
        gb.configure_column('input', header_name='User Input')
        gb.configure_column(
            'output',
            header_name='Response',
        )
        gb.configure_column('total_tokens', header_name='Total Tokens')
        gb.configure_column('total_cost', header_name='Total Cost')
        gb.configure_column('tags', header_name='Tags')
        gb.configure_column('ts', header_name='Time Stamp')

        for feedback_col in evaluations_df.columns.drop(['chain_id', 'ts',
                                                         'total_tokens',
                                                         'total_cost']):
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

        if len(selected_rows) == 0:
            st.write("Hint: select a row to display chain metadata")

        else:
            st.header(f"Selected Chain ID: {selected_rows['chain_id'][0]}")
            st.text(f"Selected Record ID: {selected_rows['record_id'][0]}")
            prompt = selected_rows['input'][0]
            response = selected_rows['output'][0]
            with st.expander("Input Prompt", expanded=True):
                st.write(prompt)

            with st.expander("Response", expanded=True):
                st.write(response)

            record_str = selected_rows['record_json'][0]
            record_json = json.loads(record_str)

            details = selected_rows['chain_json'][0]
            details_json = json.loads(details)
            #json.loads(details))  # ???

            chain_json = details_json['chain']

            llm_queries = list(
                TruDB.matching_objects(
                    details_json,
                    match=lambda q, o: len(q._path) > 0 and "llm" == q._path[-1]
                )
            )

            prompt_queries = list(
                TruDB.matching_objects(
                    details_json,
                    match=lambda q, o: len(q._path) > 0 and "prompt" == q._path[
                        -1] and "_call" not in q._path
                )
            )

            max_len = max(len(llm_queries), len(prompt_queries))

            for i in range(max_len):
                if i < len(llm_queries):
                    query, llm_details_json = llm_queries[i]
                    path_str = TruDB._query_str(query)
                    st.header(f"Chain Step {i}: {path_str.replace('.llm', '')}")
                    st.subheader(f"LLM Details:")

                    llm_kv = {
                        k: v
                        for k, v in llm_details_json.items()
                        if (v is not None) and not is_empty(v) and
                        not is_noserio(v)
                    }
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                thead tr th:first-child {display:none}
                                tbody th {display:none}
                                </style>
                                """
                    df = pd.DataFrame.from_dict(llm_kv, orient='index')
                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(df.transpose())

                if i < len(prompt_queries):
                    query, prompt_details_json = prompt_queries[i]
                    path_str = TruDB._query_str(query)
                    st.subheader(f"Prompt Details:")

                    prompt_types = {
                        k: v
                        for k, v in prompt_details_json.items()
                        if (v is not None) and not is_empty(v) and
                        not is_noserio(v)
                    }

                    for key, value in prompt_types.items():
                        with st.expander(key.capitalize(), expanded=True):
                            if isinstance(value, (Dict, List)):
                                st.write(value)
                            else:
                                if isinstance(value, str) and len(value) > 32:
                                    st.text(value)
                                else:
                                    st.write(value)
            st.header("More options:")
            if st.button("Display full chain json"):

                st.write(details_json)

            if st.button("Display full record json"):

                st.write(record_json)

    with tab2:
        feedback = feedback_cols
        cols = 4
        rows = len(feedback) // cols + 1

        for row_num in range(rows):
            with st.container():
                columns = st.columns(cols)
                for col_num in range(cols):
                    with columns[col_num]:
                        ind = row_num * cols + col_num
                        if ind < len(feedback):
                            # Generate histogram
                            fig, ax = plt.subplots()
                            bins = [
                                0, 0.2, 0.4, 0.6, 0.8, 1.0
                            ]  # Quintile buckets
                            ax.hist(
                                chain_df[feedback[ind]],
                                bins=bins,
                                edgecolor='black',
                                color='#2D736D'
                            )
                            ax.set_xlabel('Feedback Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title(feedback[ind], loc='center')
                            st.pyplot(fig)
