import streamlit as st
import pandas as pd
import numpy as np
import tru_db
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder



lms = tru_db.LocalModelStore()
df, df_feedback = lms.get_records_and_feedback([])

if 'chain' in st.session_state:
    model = st.session_state.chain
else:
    model = None

    models = list(df.model_id.unique())

    
options = st.multiselect('Choose a model', models, default = model)

# col0, col1, col2, col3 = st.columns(4)
# model_df = df.loc[df.model_id.isin(options)]
# col0.metric("Name", model)
# col1.metric("Records", len(model_df))
# col2.metric("Cost", "$0.43")
# col3.metric("Feedback Results", len(df_feedback.columns))

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

#mds = model_store.ModelDataStore()

with tab2:
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_pagination()
    gb.configure_side_bar()
    #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    data = AgGrid(df, gridOptions=gridOptions)

