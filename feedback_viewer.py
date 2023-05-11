import sqlite3
import time

import pandas as pd
import streamlit as st

import tru

lms = tru.LocalModelStore()

df, df_feedback = lms.get_records_and_feedback([])
st.dataframe(df)
st.dataframe(df_feedback)

