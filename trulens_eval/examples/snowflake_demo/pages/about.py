import streamlit as st

from common_ui import page_setup

page_setup("About")

st.info(
    """**TLDR:** Click "Chat" on the left, play around, then come back here to see what you missed.""",
    icon="👈",
)

DESCRIPTION = """
### Overview

This app enables conversation with several LLMs under various configurations,
along with simple human feedback and persisted conversation history.

Log in to save your conversations. Use Admin Mode to manage users, view
aggregated feedback stats, as well as view automated evaluation _(coming soon!)_.

### Streamlit feature tour

This app also highlights many recent features in Streamlit, and what's possible
in an integrated LLM application (less than ~1000 LOC, pure native streamlit).
To get a full tour, you should try the following:

- Visit "Chat" to configure one or two models and have a conversation with the LLMs.
  - Click the "Record feedback" button in your conversation and add some feedback.
- Visit "My Account" to view your stats, load previous conversations or export.
- Logout via "Settings", then log back in with "Admin mode" and see the nav changes.
- Visit "User Management" to see stats for other users.
- Finally, visit "Conversation Analysis" to see aggregated data and drill down.
  - You can select a bar in the chart to drilldown into a subset of conversations.
  - You can also select a single conversation in the table for full details.

### Contributing

Want to make this app even more awesome? We're looking to add more conversation data
and we'd love your help. Once you finish your tour, take a few minutes to:

- Log in as a new user (feel free to use your first name or pseudonym)
- Have several conversations with various configurations, use the existing categories
  from the Feedback dialog as a guide for topics.
  - **NOTE:** Anything you say or any LLM response will be shared publicly!
  - Make sure you add feedback for the LLM for most of your conversations.
  - Adding feedback or hitting "Save conversation" in settings will record the current
    state.
- After you finish, navigate to "My Account" and hit "Export conversations" to copy
  or download your conversations.
- Send them to Joshua (on Slack or via email `joshua.carroll@snowflake.com`)
"""

st.markdown(DESCRIPTION)
