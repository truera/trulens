import base64

import pkg_resources
import streamlit as st


def add_logo():
    logo = open(
        pkg_resources.resource_filename(
            'trulens_eval', 'ux/trulens_logo.svg'
        ), "rb"
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
        </style>
        """,
        unsafe_allow_html=True,
    )
