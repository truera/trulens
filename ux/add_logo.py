import base64

import streamlit as st


def add_logo():
    logo = open("trulens_logo.png", "rb").read()
    logo_encoded = base64.b64encode(logo).decode()
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{logo_encoded}');
                background-repeat: no-repeat;
                background-size: 300px auto;
                padding-top: 250px;
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
