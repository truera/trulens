import math

from millify import millify
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.runtime.legacy_caching.clear_cache()

from trulens_eval import Tru
from trulens_eval import tru_db

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_eval.ux.add_logo import add_logo

add_logo()

tru = Tru()
lms = tru.db


def app():
    # Set the title and subtitle of the app
    st.title('Chain Leaderboard')
    st.write(
        'Average feedback values displayed in the range from 0 (worst) to 1 (best).'
    )
    df, feedback_col_names = lms.get_records_and_feedback([])

    if df.empty:
        st.write("No records yet...")
        return

    df = df.sort_values(by="chain_id")

    if df.empty:
        st.write("No records yet...")

    chains = list(df.chain_id.unique())
    st.markdown("""---""")

    for chain in chains:
        st.header(chain)
        col1, col2, col3, *feedback_cols, col99 = st.columns(
            4 + len(feedback_col_names)
        )
        chain_df = df.loc[df.chain_id == chain]
        #model_df_feedback = df.loc[df.chain_id == model]

        col1.metric("Records", len(chain_df))
        col2.metric(
            "Cost",
            f"${millify(round(sum(cost for cost in chain_df.total_cost if cost is not None), 5), precision = 2)}"
        )
        col3.metric(
            "Tokens",
            millify(
                sum(
                    tokens for tokens in chain_df.total_tokens
                    if tokens is not None
                ),
                precision=2
            )
        )

        for i, col_name in enumerate(feedback_col_names):
            mean = chain_df[col_name].mean()

            if i < len(feedback_cols):
                if math.isnan(mean):
                    pass

                else:
                    feedback_cols[i].metric(col_name, round(mean, 2))

            else:
                if math.isnan(mean):
                    pass

                else:
                    feedback_cols[i].metric(col_name, round(mean, 2))

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
