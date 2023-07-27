import json
import math

from millify import millify
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.ux.styles import CATEGORY

st.runtime.legacy_caching.clear_cache()

from trulens_eval import db
from trulens_eval import Tru
from trulens_eval.ux import styles
from trulens_eval.ux.components import draw_metadata

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_eval.ux.add_logo import add_logo

add_logo()

tru = Tru()
lms = tru.db


def streamlit_app():
    # Set the title and subtitle of the app
    st.title('App Leaderboard')
    st.write(
        'Average feedback values displayed in the range from 0 (worst) to 1 (best).'
    )
    df, feedback_col_names = lms.get_records_and_feedback([])

    if df.empty:
        st.write("No records yet...")
        return

    df = df.sort_values(by="app_id")

    if df.empty:
        st.write("No records yet...")

    apps = list(df.app_id.unique())
    st.markdown("""---""")

    for app in apps:
        app_df = df.loc[df.app_id == app]
        if app_df.empty:
            continue
        app_str = app_df['app_json'].iloc[0]
        app_json = json.loads(app_str)
        metadata = app_json.get('metadata')
        #st.text('Metadata' + str(metadata))
        st.header(app, help=draw_metadata(metadata))
        col1, col2, col3, col4, *feedback_cols, col99 = st.columns(
            5 + len(feedback_col_names)
        )
        latency_mean = app_df['latency'].apply(
            lambda td: td if td != MIGRATION_UNKNOWN_STR else None
        ).mean()

        #app_df_feedback = df.loc[df.app_id == app]

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            f"{millify(round(latency_mean, 5), precision=2)}"
            if not math.isnan(latency_mean) else "nan"
        )
        col3.metric(
            "Total Cost (USD)",
            f"${millify(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision = 2)}"
        )
        col4.metric(
            "Total Tokens",
            millify(
                sum(
                    tokens for tokens in app_df.total_tokens
                    if tokens is not None
                ),
                precision=2
            )
        )
        for i, col_name in enumerate(feedback_col_names):
            mean = app_df[col_name].mean()

            st.write(
                styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            if math.isnan(mean):
                pass

            else:
                cat = CATEGORY.of_score(mean)
                feedback_cols[i].metric(
                    label=col_name,
                    value=f'{round(mean, 2)}',
                    delta=f'{cat.icon} {cat.adjective}',
                    delta_color="normal"
                    if mean >= CATEGORY.PASS.threshold else "inverse"
                )

        with col99:
            if st.button('Select App', key=f"app-selector-{app}"):
                st.session_state.app = app
                switch_page('Evaluations')

        #with st.expander("Model metadata"):
        #    st.markdown(draw_metadata(metadata))

        st.markdown("""---""")


# Define the main function to run the app
def main():
    streamlit_app()


if __name__ == '__main__':
    main()
