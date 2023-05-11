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




# Define the main function to run the app
def main():
    app()

if __name__ == '__main__':
    main()
