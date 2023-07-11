import base64

import pkg_resources
import streamlit as st

from trulens_eval import __package__
from trulens_eval import __version__


def add_logo():
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
            }}
            [data-testid="stSidebarNav"]::before {{
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}
            [data-testid="stSidebarNav"]::after {{
                margin-left: 20px;
                color: #aaaaaa;
                content: "{__package__} {__version__}";
                font-size: 10pt;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
