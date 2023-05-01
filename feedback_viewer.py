import sqlite3
import streamlit as st
import time
import pandas as pd

# Set up SQLite database connection
conn = sqlite3.connect("llm_quality.db")
c = conn.cursor()

# Set up Streamlit app
st.title("Feedback Viewer")

# Define function to show table contents
def show_table_contents(table_name):
    c.execute(f"SELECT * FROM {table_name}")
    rows = c.fetchall()
    if len(rows) == 0:
        st.write("Table is empty.")
    else:
        df = pd.DataFrame(rows, columns=[description[0] for description in c.description])
        st.dataframe(df)


table_name = "llm_calls"



show_table_contents(table_name)
