from dotenv import load_dotenv

load_dotenv()

from typing import List

from common_ui import chat_response
from common_ui import configure_model
from common_ui import generate_title
from common_ui import page_setup
from common_ui import st_thread
from conversation_manager import ConversationManager
from schema import Conversation
from schema import ConversationFeedback
from schema import ConversationRecord
from schema import Message
import streamlit as st

title = "Chat"
if st.session_state.get("conversation_title"):
    title = f"Chat: {st.session_state.conversation_title}"

sidebar_container = page_setup(
    title=title,
    wide_mode=True,
    collapse_sidebar=False,
)

DEFAULT_MESSAGE = "Hello there! Let's chat?"
MODELS_HELP_STR = "Select an available model"

conv_mgr: ConversationManager = st.session_state.conversation_manager


def save_conversation():
    cr = ConversationRecord(
        title=st.session_state.get("conversation_title"),
        user=st.session_state.get("user_name"),
        conversations=st.session_state["conversations"],
    )
    conv_mgr.add_or_update(
        cr, persist=st.secrets.get("enablePersistence", True)
    )


# Handle case where we navigated to load an existing conversation:
if to_load := st.session_state.pop("load_conversation", None):
    cr = conv_mgr.get_by_title(to_load)
    st.session_state["conversations"] = cr.conversations
    st.session_state["conversation_title"] = cr.title
    st.rerun()


# Handle changing from single or multi model mode
def update_model_mode():
    if st.session_state.multi_mode:
        st.session_state["conversations"] = [Conversation(), Conversation()]
    else:
        st.session_state["conversations"] = [Conversation()]
    for conversation in st.session_state["conversations"]:
        conversation.add_message(
            Message(role="assistant", content=DEFAULT_MESSAGE), render=False
        )


with st.sidebar:
    enable_toggle = (
        "conversations" in st.session_state and
        len(st.session_state.conversations[0].messages) > 2
    )
    # st.toggle(
    #     "Multi-model mode",
    #     disabled=enable_toggle,
    #     key="multi_mode",
    #     on_change=update_model_mode,
    # )

# Store conversation state in streamlit session
if "conversations" not in st.session_state:
    st.session_state["conversations"] = [Conversation()]
    for conversation in st.session_state["conversations"]:
        conversation.add_message(
            Message(role="assistant", content=DEFAULT_MESSAGE), render=False
        )
conversations: List[Conversation] = st.session_state["conversations"]

# Main area
""

# Render model config control
if len(conversations) == 1:
    config_col_spec = [7, 3]
else:
    config_col_spec = [5, 5]
model_config_cols = st.columns(config_col_spec)
for idx, conversation in enumerate(conversations):
    conversation.model_config = configure_model(
        container=model_config_cols[idx],
        model_config=conversation.model_config,
        key=f"{idx}",
    )

# Handle any dangling errors from previous runs
# (Doing it here means toasts will show up correctly)
if any([c.has_error for c in conversations]):
    st.toast(
        "Something went wrong while generating a response. See logs for details.",
        icon=":material/error:",
    )
    for c in conversations:
        if len(c.messages[-1].content) == 0:
            c.messages[
                -1
            ].content = "Something went wrong while generating a response."
        c.has_error = None


def render_msgs():
    # Render the chat
    for idx, msg in enumerate(conversations[0].messages):
        if msg.role == "user":
            conversations[0].render_message(msg, key=str(idx))
        else:
            msg_cols = st.columns(len(conversations))
            for i, conversation in enumerate(conversations):
                conversation.render_message(
                    conversation.messages[idx],
                    container=msg_cols[i],
                    key=f"conversation_{i}_idx_{idx}"
                )


render_msgs()

user_msg = st.empty()
response = st.empty()
response_controls = st.empty()

user_input = st.chat_input("Enter your message here."
                          ) or st.session_state.pop("regenerate", None)
if user_input:
    new_msg = Message(role="user", content=user_input)
    for c in conversations:
        c.add_message(new_msg, render=False)
    conversations[0].render_message(
        new_msg, container=user_msg, key=str(len(conversations[0].messages))
    )

    msg_cols = response.columns(len(conversations))
    threads = []
    for i, conversation in enumerate(conversations):
        args = (
            conversation,
            msg_cols[i],
        )
        t = st_thread(target=chat_response, args=args)
        threads.append(t)
        t.start()

    if "conversation_title" not in st.session_state:
        title_dict = dict()
        t = st_thread(
            target=generate_title,
            args=(user_input, title_dict, conversation.model_config)
        )
        t.start()
        t.join()
        if "output" in title_dict:
            st.session_state.conversation_title = title_dict["output"]

    for t in threads:
        t.join()
    st.rerun()  # Clear stale containers

# Add action buttons


@st.experimental_dialog("Record feedback")
def record_feedback():
    TOPIC_CATEGORIES = ["", "Technical", "Travel", "Personal advice", "Other"]
    # Only give feedback on one model config if multiple exist
    if len(conversations) == 2:
        model_choices = [
            f"Left: `{conversations[0].model_config.model}`",
            f"Right: `{conversations[1].model_config.model}`",
        ]
        model = st.radio(
            "Which model response are you providing feedback on?", model_choices
        )
        conv_idx = model_choices.index(model)
    else:
        conv_idx = 0

    # Support pre-populating existing values from earlier feedback
    vals = {}
    if existing := conversations[conv_idx].feedback:
        vals = dict(
            category_idx=TOPIC_CATEGORIES.index(existing.category),
            custom_category=existing.custom_category,
            score=int(existing.quality_score * 10),
            comments=existing.comments,
            flagged=existing.flagged,
            flagged_comments=existing.flagged_comments,
        )

    if "conversation_title" not in st.session_state:
        title_warning = st.empty()
        title = st.text_input(
            "Conversation title:", help="Add a short, descriptive title"
        )
    category_warning = st.empty()
    category = st.selectbox(
        "Topic category:", TOPIC_CATEGORIES, index=vals.get("category_idx")
    )
    if category == "Other":
        custom_category = st.text_input(
            "Custom category:", value=vals.get("custom_category")
        )

    SCORE_HELP = """
    Enter a score on the quality score. 7-8 indicates a person knowledgeable in the
    topic would be satisfied with the responses. Less than 5 indicates responses
    that are actively incorrect, anti-helpful and/or harmful.
    """
    score = st.slider(
        "Quality score:",
        0,
        10,
        step=1,
        help=SCORE_HELP,
        value=vals.get("score") or 5
    )
    comments = st.text_input("Comments:", value=vals.get("comments"))
    if not vals.get("flagged") and score < 3:
        vals["flagged"] = True
    flagged = st.checkbox("Flag for review", value=vals.get("flagged"))
    if flagged:
        flagged_comments = st.text_input(
            "Why is this flagged?", value=vals.get("flagged_comments")
        )

    if not st.session_state.get("user_name"):
        st.warning(
            "Please login to persist your feedback", icon=":material/warning:"
        )
    if st.button("Submit"):
        if not category:
            category_warning.warning(
                "Category is required", icon=":material/warning:"
            )
            st.stop()
        feedback = ConversationFeedback(
            category=category,
            quality_score=score / 10.0,
            flagged=flagged,
        )
        if comments:
            feedback.comments = comments
        if flagged and flagged_comments:
            feedback.flagged_comments = flagged_comments
        if category == "Other":
            feedback.custom_category = custom_category
        conversations[conv_idx].feedback = feedback
        if st.session_state.get("user_name"):
            if "conversation_title" not in st.session_state:
                if title:
                    st.session_state.conversation_title = title
                else:
                    title_warning.warning(
                        "Title is required", icon=":material/warning:"
                    )
                    st.stop()
            save_conversation()
            st.session_state["pending_feedback"] = True
        st.rerun()


def clear_conversation():
    for conversation in conversations:
        conversation.reset_messages()
        conversation.add_message(
            Message(role="assistant", content=DEFAULT_MESSAGE), render=False
        )
        conversation.feedback = None
    st.session_state.pop("conversation_title", None)
    st.toast("Started new chat", icon=":material/edit_square:")


def regenerate():
    st.session_state.regenerate = conversations[0].messages[-2].content
    for conversation in conversations:
        del conversation.messages[-2:]


@st.experimental_dialog("Edit conversation title")
def edit_title():
    new_title = st.text_input(
        "New conversation title:",
        value=st.session_state.get("conversation_title"),
    )
    if st.button("Save"):
        st.session_state.conversation_title = new_title
        st.rerun()


if len(conversations[0].messages) > 1:
    action_cols = response_controls.columns(3)

    action_cols[0].button(
        "🔄&nbsp; Regenerate", use_container_width=True, on_click=regenerate
    )
    action_cols[1].button(
        "📝&nbsp; New chat",
        use_container_width=True,
        on_click=clear_conversation,
    )
    if action_cols[2].button(
            "📬&nbsp; Add feedback",
            use_container_width=True,
    ):
        record_feedback()

    if st.session_state.get("user_name"):
        with sidebar_container:
            if st.button("✏️&nbsp; Edit title", use_container_width=True):
                edit_title()
            if "user_name" in st.session_state and "conversation_title" in st.session_state:
                if st.button("💾&nbsp; Save conversation",
                             use_container_width=True):
                    save_conversation()
                    st.toast(
                        "Conversation saved successfully",
                        icon=":material/check_circle:"
                    )

if st.session_state.pop("pending_feedback", None):
    st.toast("Feedback submitted successfully", icon=":material/rate_review:")
