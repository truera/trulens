import sqlite3
import time

import pandas as pd
import streamlit as st

# Set up SQLite database connection
conn = sqlite3.connect("llm_quality.db")
c = conn.cursor()

# Set up Streamlit app
st.title("Feedback Viewer")


# Define function to show table contents
def show_table_contents(logging_table_name, feedback_table_name):
    c.execute(
        f"SELECT l.*, f.feedback_functions FROM {logging_table_name} l LEFT JOIN {feedback_table_name} f on l.record_id = f.record_id"
    )
    rows = c.fetchall()
    if len(rows) == 0:
        st.write("Table is empty.")
    else:
        df = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )
        st.dataframe(df)


logging_table_name = "llm_calls"
feedback_table_name = "llm_feedback_functions"

show_table_contents(logging_table_name, feedback_table_name)
