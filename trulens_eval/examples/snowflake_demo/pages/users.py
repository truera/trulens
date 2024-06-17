from common_ui import page_setup
from conversation_manager import ConversationManager
import streamlit as st

page_setup("User Management", visibility="admin")

st.caption("View basic stats for existing users. More features coming soon.")

conv_mgr: ConversationManager = st.session_state.conversation_manager
users = conv_mgr.list_users()

for user in users:
    st.subheader(f"User: {user}")
    conversations = conv_mgr.get_all_conversations(user=user)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Total conversations x models", len(conversations))
    metric_cols[1].metric(
        "Total feedback", len([c for c in conversations if c.feedback])
    )
