import math

import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.runtime.legacy_caching.clear_cache()

from trulens_evalchain import tru_db

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_evalchain.ux.add_logo import add_logo

add_logo()

#import model_store

#@st.cache_resource
#def get_model_store():
#    return model_store.ModelDataStore()

lms = tru_db.LocalSQLite()


def app():
    # Set the title and subtitle of the app
    st.title('Chain Leaderboard')
    df, df_feedback = lms.get_records_and_feedback([])

    if df.empty:
        st.write("No records yet...")

    models = list(df.chain_id.unique())
    st.markdown("""---""")

    for model in models:
        col0, col1, col2, col3, *feedback_cols, col99 = st.columns(
            5 + len(df_feedback.columns)
        )
        model_df = df.loc[df.chain_id == model]
        model_df_feedback = df_feedback.loc[df.chain_id == model]

        col0.metric("Name", model)
        col1.metric("Records", len(model_df))
        col2.metric(
            "Cost",
            round(
                sum(cost for cost in model_df.total_cost if cost is not None), 5
            )
        )
        col3.metric(
            "Tokens",
            sum(
                tokens for tokens in model_df.total_tokens if tokens is not None
            )
        )

        for i, col_name in enumerate(df_feedback.columns):
            mean = model_df_feedback[col_name].mean()

            if i < len(feedback_cols):
                if math.isnan(mean):
                    pass

                elif mean < 0.5:
                    feedback_cols[i].metric(
                        col_name,
                        round(mean, 2),
                        delta="Fail",
                        delta_color="inverse"
                    )
                else:
                    feedback_cols[i].metric(
                        col_name,
                        round(mean, 2),
                        delta="Pass",
                        delta_color="normal"
                    )
            else:
                if math.isnan(mean):
                    pass

                elif mean < 0.5:
                    feedback_cols[i].metric(
                        col_name,
                        round(mean, 2),
                        delta="Fail",
                        delta_color="inverse"
                    )
                else:
                    feedback_cols[i].metric(
                        col_name,
                        round(mean, 2),
                        delta="Pass",
                        delta_color="normal"
                    )

        with col99:
            if st.button('Select Chain', key=f"model-selector-{model}"):
                st.session_state.chain = model
                switch_page('Evaluations')

        st.markdown("""---""")


# Define the main function to run the app
def main():
    app()


if __name__ == '__main__':
    main()
