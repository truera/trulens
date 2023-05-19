import json

import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import streamlit as st
from ux.add_logo import add_logo

from trulens_eval import tru_db

st.set_page_config(page_title="Evaluations [generic]", layout="wide")

st.title("Evaluations [generic]")

st.runtime.legacy_caching.clear_cache()

add_logo()

lms = tru_db.LocalTinyDB("slackbot.json")

df, df_feedback = lms.get_records_and_feedback([])

if df.empty:
    st.write("No records yet...")

else:

    if 'Chains' in st.session_state:
        chain = st.session_state.chain
    else:
        chain = None

    chains = list(df.chain_id.unique())

    options = st.multiselect('Filter Chains', chains, default=chain)

    chain_df = df.loc[df.chain_id.isin(options)]
    chain_df_feedback = df_feedback.loc[df_feedback.chain_id.isin(options)]

    if (len(options) == 0):
        st.header("All Chains")
    elif (len(options) == 1):
        st.header(options[0])
    else:
        st.header("Multiple Chains Selected")

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:
        evaluations_df = df

        # print("df=")
        # print(df)

        # print("feedback=")
        # print(df_feedback)

        evaluations_df = evaluations_df.merge(
            on=['record_id', 'chain_id'], right=df_feedback, how="outer"
        )

        # print("merged=")
        # print(evaluations_df)

        evaluations_df_str = evaluations_df.copy()
        for col in evaluations_df_str.columns:
            evaluations_df_str[col] = evaluations_df_str[col].map(
                lambda v: str(v) if isinstance(v, dict) else v
            )

        gb = GridOptionsBuilder.from_dataframe(evaluations_df_str)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)

        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df_str,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED
        )

        selected_rows = data['selected_rows']
        selected_rows = evaluations_df[evaluations_df.record_id.isin(
            map(lambda r: r['record_id'], selected_rows)
        )]

        if len(selected_rows) != 0:
            for i, row in selected_rows.iterrows():
                st.json(row.record)
        else:
            st.write("Hint: select a row to display chain metadata")

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
                            #st.text(feedback[ind])
                            print(chain_df_feedback[feedback[ind]])
                            #st.bar_chart(chain_df_feedback[feedback[ind]])
