import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from trulens_eval.schema import Record
from trulens_eval.util import GetItemOrAttribute
from ux.add_logo import add_logo

import streamlit.components.v1 as components

from trulens_eval import Tru
from trulens_eval import tru_db
from trulens_eval.util import is_empty, matching_objects
from trulens_eval.util import is_noserio
from trulens_eval.tru_db import TruDB
from trulens_eval.ux.components import draw_calls
from trulens_eval.ux.styles import cellstyle_jscode
from trulens_eval.tru_feedback import default_pass_fail_color_threshold

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
    if 'chain' in st.session_state:
        chain = st.session_state.chain
    else:
        chain = chains

    options = st.multiselect('Filter Applications', chains, default=chain)

    if (len(options) == 0):
        st.header("All Applications")
        chain_df = df_results

    elif (len(options) == 1):
        st.header(options[0])

        chain_df = df_results[df_results.chain_id.isin(options)]

    else:
        st.header("Multiple Applications Selected")

        chain_df = df_results[df_results.chain_id.isin(options)]

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:
        gridOptions = {'alwaysShowHorizontalScroll': True}
        evaluations_df = chain_df
        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        cellstyle_jscode = JsCode(cellstyle_jscode)

        gb.configure_column('record_json', header_name='Record JSON', hide=True)
        gb.configure_column('chain_json', header_name='Chain JSON', hide=True)
        gb.configure_column('cost_json', header_name='Cost JSON', hide=True)

        gb.configure_column('record_id', header_name='Record ID', hide=True)
        gb.configure_column('chain_id', header_name='Chain ID')
        gb.configure_column('feedback_id', header_name='Feedback ID', hide=True)
        gb.configure_column('input', header_name='User Input')
        gb.configure_column(
            'output',
            header_name='Response',
        )
        gb.configure_column('total_tokens', header_name='Total Tokens (#)')
        gb.configure_column('total_cost', header_name='Total Cost (USD)')
        gb.configure_column('tags', header_name='Tags')
        gb.configure_column('ts', header_name='Time Stamp')

        for feedback_col in evaluations_df.columns.drop(['chain_id', 'ts',
                                                         'total_tokens',
                                                         'total_cost']):
            gb.configure_column(
                feedback_col,
                cellStyle=cellstyle_jscode,
                hide=feedback_col.endswith("_calls")
            )

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
            st.header(
                f"Selected LLM Application: {selected_rows['chain_id'][0]}"
            )
            st.text(f"Selected Record ID: {selected_rows['record_id'][0]}")

            prompt = selected_rows['input'][0]
            response = selected_rows['output'][0]

            with st.expander("Input Prompt", expanded=True):
                st.write(prompt)

            with st.expander("Response", expanded=True):
                st.write(response)

            row = selected_rows.head().iloc[0]

            st.header("Feedback")
            for fcol in feedback_cols:
                feedback_name = fcol
                feedback_result = row[fcol]
                feedback_calls = row[f"{fcol}_calls"]

                def display_feedback_call(call):

                    def highlight(s):
                        return ['background-color: #4CAF50'] * len(
                            s
                        ) if s.result >= default_pass_fail_color_threshold else [
                            'background-color: #FCE6E6'
                        ] * len(s)

                    if (len(call) > 0):
                        df = pd.DataFrame.from_records(
                            [call[i]["args"] for i in range(len(call))]
                        )
                        df["result"] = pd.DataFrame(
                            [float(call[i]["ret"]) for i in range(len(call))]
                        )
                        st.dataframe(
                            df.style.apply(highlight, axis=1
                                          ).format("{:.2}", subset=["result"])
                        )
                    else:
                        st.text("No feedback details.")

                with st.expander(f"{feedback_name} = {feedback_result}",
                                 expanded=True):
                    display_feedback_call(feedback_calls)

            record_str = selected_rows['record_json'][0]
            record_json = json.loads(record_str)
            record = Record(**record_json)

            details = selected_rows['chain_json'][0]
            chain_json = json.loads(
                details
            )  # chains may not be deserializable, don't try to, keep it json.

            step_llm = GetItemOrAttribute(item_or_attribute="llm")
            step_prompt = GetItemOrAttribute(item_or_attribute="prompt")
            step_call = GetItemOrAttribute(item_or_attribute="_call")

            llm_queries = list(
                matching_objects(
                    chain_json,
                    match=lambda q, o: len(q.path) > 0 and step_llm == q.path[-1
                                                                             ]
                )
            )

            prompt_queries = list(
                matching_objects(
                    chain_json,
                    match=lambda q, o: len(q.path) > 0 and step_prompt == q.
                    path[-1] and step_call not in q._path
                )
            )

            max_len = max(len(llm_queries), len(prompt_queries))

            for i in range(max_len + 1):
                st.header(f"Component {i+1}")
                draw_calls(record, index=i + 1)

                if i < len(llm_queries):
                    query, llm_details_json = llm_queries[i]
                    st.subheader(f"LLM Details:")
                    path_str = str(query)
                    st.text(path_str[:-4])

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
                    df = pd.DataFrame.from_dict(
                        llm_kv, orient='index'
                    ).transpose()

                    # Iterate over each column of the DataFrame
                    for column in df.columns:
                        # Check if any cell in the column is a dictionary
                        if any(isinstance(cell, dict) for cell in df[column]):
                            # Create new columns for each key in the dictionary
                            new_columns = df[column].apply(
                                lambda x: pd.Series(x)
                                if isinstance(x, dict) else pd.Series()
                            )
                            new_columns.columns = [
                                f"{key}" for key in new_columns.columns
                            ]

                            # Remove extra zeros after the decimal point
                            new_columns = new_columns.applymap(
                                lambda x: '{0:g}'.format(x)
                                if isinstance(x, float) else x
                            )

                            # Add the new columns to the original DataFrame
                            df = pd.concat(
                                [df.drop(column, axis=1), new_columns], axis=1
                            )
                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(df)

                if i < len(prompt_queries):
                    query, prompt_details_json = prompt_queries[i]
                    path_str = str(query)
                    st.subheader(f"Prompt Details:")
                    st.text(path_str)

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

                st.write(chain_json)

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
