import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.switch_page_button import switch_page
import tru_db
#import model_store

#@st.cache_resource
#def get_model_store():
#    return model_store.ModelDataStore()

lms = tru_db.LocalModelStore()

# Create a sample dataframe
#data = {'Name': ['Chain A', 'Chain B', 'Chain C', 'Chain D', 'Chain E'],
#        'Model Type': ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5'],
#        'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']}
#df = pd.DataFrame(data)

# Define the layout of the app
def app():
    # Set the title and subtitle of the app
    st.title('Model Leaderboard')
    #st.subheader('First Page')
    df, df_feedback = lms.get_records_and_feedback([])
    models = list(df.model_id.unique())

    for model in models:
        col0, col1, col2, col3, col4 = st.columns(5)
        model_df = df.loc[df.model_id == model]

        col0.metric("Name", model)
        col1.metric("Records", len(model_df))
        col2.metric("Cost", "$0.43")
        col3.metric("Feedback Results", len(df_feedback.columns))
    with col4:
            if st.button('Select Chain', key = f"model-selector-{model}"):
                st.session_state.chain = model
                switch_page('Chain')

    # # Display the items as columns with a button to select a row
    # cols = st.columns(4)
    # headers = ["Chain ID", "Model Type", "Date Created", ""]
    # for col, header in zip(cols, headers):
    #     with col:
    #         st.subheader(header)

    # for i, row in df.iterrows():
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         st.write(row['Name'])
    #     with col2:
    #         st.write(row['Model Type'])
    #     with col3:
    #         st.write(row['Date'])
    #     with col4:
    #         if st.button('Select chain', key = f"model-selector-{i}"):
    #             st.session_state.chain = i
    #             switch_page('Chain')


# Define the main function to run the app
def main():
    app()

if __name__ == '__main__':
    main()
