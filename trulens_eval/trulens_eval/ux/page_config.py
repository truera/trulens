import base64

import streamlit as st

from trulens_eval import __package__
from trulens_eval import __version__
from trulens_eval.utils.imports import static_resource


def set_page_config(page_title="TruLens"):

    st.set_page_config(
        page_title=page_title,
        page_icon="https://www.trulens.org/img/favicon.ico",
        layout="wide"
    )

    logo = static_resource("ux/trulens_logo.svg")

    st.logo(str(logo), link="https://www.trulens.org/")

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-repeat: no-repeat;
                background-size: 300px auto;
                background-position: 20px 20px;
                height: calc(100vh - 80px);
            }}
            [data-testid="stSidebarNav"]::before {{
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}

            /* For user feedback button and version text */
            [data-testid="stSidebarUserContent"] {{
                padding-bottom: 1rem;
            }}

            [data-testid="stSidebarUserContent"] [data-testid="column"] {{
                align-content: center;
            }}

            [data-testid="stSidebarUserContent"] [data-testid="stText"] {{
                color: #aaaaaa;
                font-size: 9pt;
            }}

            /* For list items in st.dataframe */
            #portal .clip-region .boe-bubble {{
                height: auto;
                border-radius: 4px;
                padding: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar feedback button
    with st.sidebar:
        version_col, user_feedback_col = st.columns(2)
        with version_col:
            st.text(f"{__package__}\nv{__version__}")
        with user_feedback_col:
            st.link_button(
                "Share Feedback",
                "https://forms.gle/HAc4HBk5nZRpgw7C6",
                help="Help us improve TruLens!"
            )
