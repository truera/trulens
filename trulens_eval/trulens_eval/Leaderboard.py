import math

import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.runtime.legacy_caching.clear_cache()

from trulens_eval import tru_db

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_eval.ux.add_logo import add_logo

add_logo()

#import model_store

#@st.cache_resource
#def get_model_store():
#    return model_store.ModelDataStore()

lms = tru_db.LocalSQLite()


def app():
    # Set the title and subtitle of the app
    st.title('Chain Leaderboard')
    df, feedback_col_names = lms.get_records_and_feedback([])

    df = df.sort_values(by="chain_id")

    if df.empty:
        st.write("No records yet...")

    chains = list(df.chain_id.unique())
    st.markdown("""---""")

    for chain in chains:
        col0, col1, col2, col3, *feedback_cols, col99 = st.columns(
            5 + len(feedback_col_names)
        )
        chain_df = df.loc[df.chain_id == chain]
        #model_df_feedback = df.loc[df.chain_id == model]

        col0.metric("Name", chain)
        col1.metric("Records", len(chain_df))
        col2.metric(
            "Cost",
            round(
                sum(cost for cost in chain_df.total_cost if cost is not None), 5
            )
        )
        col3.metric(
            "Tokens",
            sum(
                tokens for tokens in chain_df.total_tokens if tokens is not None
            )
        )

        for i, col_name in enumerate(feedback_col_names):
            mean = chain_df[col_name].mean()

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
            if st.button('Select Chain', key=f"model-selector-{chain}"):
                st.session_state.chain = chain
                switch_page('Evaluations')

        st.markdown("""---""")


# Define the main function to run the app
def main():
    app()


if __name__ == '__main__':
    main()
