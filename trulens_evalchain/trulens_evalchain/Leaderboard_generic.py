import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.runtime.legacy_caching.clear_cache()

import tru_db

st.set_page_config(page_title="Leaderboard [generic]", layout="wide")

from ux.add_logo import add_logo

add_logo()

lms = tru_db.LocalTinyDB("slackbot.json")


def app():
    # Set the title and subtitle of the app
    st.title('Chain Leaderboard')
    df, df_feedback = lms.get_records_and_feedback([])

    if df.empty:
        st.write("No records yet...")

    chains = list(df.chain_id.unique())
    st.markdown("""---""")

    for chain in chains:
        col0, col1, col2, col3, *feedback_cols, col99 = st.columns(
            5 + len(df_feedback.columns)
        )
        chain_df = df.loc[df.chain_id == chain]
        chain_df_feedback = df_feedback.loc[df_feedback.chain_id == chain]

        col0.metric("Name", chain)
        col1.metric("Records", len(chain_df))
        # col2.metric("Cost", round(sum(chain_df.total_cost), 5))
        # col3.metric("Tokens", sum(chain_df.total_tokens))

        for row_id, feedback_dict in enumerate(df_feedback.iterrows()):
            print(feedback_dict)

        """
        for i, col_name in enumerate(df_feedback.columns):

            if i < len(feedback_cols):
                feedback_cols[i].metric(
                    col_name, round(chain_df_feedback[col_name].mean(), 2)
                )
            else:
                st.metric(
                    col_name, round(chain_df_feedback[col_name].mean(), 2)
                )
        """

        with col99:

            if st.button('Select Chain', key=f"chain-selector-{chain}"):
                st.session_state.chain = chain
                switch_page('Evaluations_generic')

        st.markdown("""---""")


# Define the main function to run the app
def main():
    app()


if __name__ == '__main__':
    main()
