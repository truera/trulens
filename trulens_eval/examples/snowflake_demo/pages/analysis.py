import altair as alt
from common_ui import page_setup
from conversation_manager import ConversationManager
import pandas as pd
import streamlit as st

page_setup("Conversation Analysis", visibility="admin")

conv_mgr: ConversationManager = st.session_state.conversation_manager
flattened = []

for record in conv_mgr._conversations:
    for conv in record.conversations:
        c = dict(
            title=record.title,
            user=record.user,
        )
        c.update(dict(conv.model_config))
        if conv.feedback:
            c.update(dict(conv.feedback))
        c["text"] = conv.messages_to_text(truncate=False)
        c["feedback"] = "Has feedback" if conv.feedback else "No feedback"
        flattened.append(c)

df = pd.DataFrame.from_records(flattened)

st.header("Feedback by model")
st.caption("Select a group to drilldown.")
c = (
    alt.Chart(df).mark_bar().encode(
        x="model",
        y="count(feedback)",
        color="feedback",
    ).add_params(alt.selection_point(name="group"))
)

chart_selection = st.altair_chart(
    c, use_container_width=True, on_select="rerun"
)

st.header("Conversation log")
if group := chart_selection.selection.group:
    selected_model = group[0]["model"]
    selected_feedback = group[0]["feedback"]
    df = df[(df["model"] == selected_model) &
            (df["feedback"] == selected_feedback)]
    st.write(
        f"Applied filters. Model: :blue[{selected_model}] &nbsp;|&nbsp; Feedback: :blue[{selected_feedback}]"
    )

df_selection = st.dataframe(df, on_select="rerun", selection_mode="single-row")
st.caption("Select a row for expanded details.")

if df_selection.selection.rows:
    st.write(flattened[df_selection.selection.rows[0]])
