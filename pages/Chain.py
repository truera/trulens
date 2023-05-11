import streamlit as st
import pandas as pd
import numpy as np
import model_store 

# import sqlite3

# class ModelDataStore:
#     def __init__(self):
#         # Set up SQLite database connection
#         conn = sqlite3.connect("llm_quality.db")
#         self.c = conn.cursor()

#     def get_all_data(self):
#         table_name = "llm_calls"
#         self.c.execute(f"SELECT * FROM {table_name}")
#         rows = self.c.fetchall()
#         if len(rows) == 0:
#             df = pd.DataFrame()
#         else:
#             df = pd.DataFrame(
#                 rows, columns=[description[0] for description in self.c.description]
#             )
#         #st.dataframe(df)
#         return df
#         #show_table_contents(table_name)


# Create a sample dataframe
data = {'Name': ['Chain A', 'Chain B', 'Chain C', 'Chain D', 'Chain E'],
        'Model Type': ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5'],
        'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']}
df = pd.DataFrame(data)

if 'chain' in st.session_state:
    chain_id = st.session_state.chain
else:
    chain_id = 0

options = st.multiselect(
    'What are your favorite colors',
    ['Chain A', 'Chain B', 'Chain C', 'Chain D', 'Chain E'])

if (len(options) == 0):
    st.header("All Chains")
else:
    st.header(options)

#st.header(df['Name'][chain_id])

col1, col2, col3 = st.columns(3)

col1.metric("Runs", "30")
col2.metric("Cost", "$0.43")
col3.metric("Model tests", "22")

tab1, tab2 = st.tabs(["Tests", "Runs"])

with tab1:
    tests = [
        'openai-moderation-response-violencegraphic',
        'openai_moderation-prompt-violencegraphic',
        'openai-text-davinci-002-response-sentiment-positive',
        'openai-text-davinci-002-prompt-sentiment-positive',
        'huggingface-twitter-roberta-response-sentiment-positive',
        'huggingface-twitter-roberta-prompt-sentiment-positive'
    ]

    cols = 4
    rows = len(tests) // cols + 1

    for row_num in range(rows):
        with st.container():
            columns = st.columns(cols)
            for col_num in range(cols):
                with columns[col_num]:
                    ind = row_num*cols+col_num
                    if ind<len(tests):
                        st.text(tests[row_num*cols+col_num])
                        st.bar_chart(np.random.randn(10, 3))

mds = model_store.ModelDataStore()

with tab2:
    st.dataframe(mds.get_all_data())

