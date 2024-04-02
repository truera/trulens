import base64

import pkg_resources
import streamlit as st

from trulens_eval import __package__
from trulens_eval import __version__


def add_logo_and_style_overrides():
    logo = open(
        pkg_resources.resource_filename('trulens_eval', 'ux/trulens_logo.svg'),
        "rb"
    ).read()

    logo_encoded = base64.b64encode(logo).decode()
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/svg+xml;base64,{logo_encoded}');
                background-repeat: no-repeat;
                background-size: 300px auto;
                padding-top: 50px;
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
        version_col, user_feedback_col = st.columns([6,4])
        with version_col:
            st.text(f"{__package__} {__version__}")
        with user_feedback_col:
            st.link_button("Feedback", "https://forms.gle/HAc4HBk5nZRpgw7C6", help="Help us improve TruLens!")
        
