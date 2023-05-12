import ast

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import tru_db

st.set_page_config(layout="wide")

#import model_store

#@st.cache_resource
#def get_model_store():
#    return model_store.ModelDataStore()

lms = tru_db.LocalModelStore()


def app():
    # Set the title and subtitle of the app
    st.title('Model Leaderboard')
    #st.subheader('First Page')
    df, df_feedback = lms.get_records_and_feedback([])
    models = list(df.model_id.unique())

    for model in models:
        col0, col1, col2, col3, col4, col5 = st.columns(6)
        model_df = df.loc[df.model_id == model]

        # Get the average feedback results
        totals = {}
        # df['feedback'] = df['feedback'].apply(lambda x: ast.literal_eval(x))
        # for row in df['feedback']:
        #     for key in row.keys():
        #         if key in totals:
        #             totals[key] += row[key]
        #         else:
        #             # If the key doesn't exist, add it to the totals dictionary
        #             totals[key] = row[key]
        # num_rows = len(df)
        # average_feedback = str(
        #     {key: totals[key] / num_rows for key in totals.keys()}
        # )
        average_feedback = "" #TODO(josh)

        col0.metric("Name", model)
        col1.metric("Records", len(model_df))
        col2.metric("Cost", sum(df.total_cost))
        col3.metric("Tokens", sum(df.total_tokens))
        col4.metric("Average Feedback", average_feedback)
    with col5:
        if st.button('Select Chain', key=f"model-selector-{model}"):
            st.session_state.chain = model
            switch_page('Chain')


# Define the main function to run the app
def main():
    app()


if __name__ == '__main__':
    main()
