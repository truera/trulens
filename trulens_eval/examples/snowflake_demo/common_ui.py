import json
import pathlib
import threading
from typing import Dict

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit_pills import pills

from conversation_manager import ConversationManager
from llm import StreamGenerator, AVAILABLE_MODELS
from schema import (
    Conversation,
    FeedbackDisplay,
    Message,
    ModelConfig,
)

# feedback functions
from feedback import feedbacks_no_rag, feedbacks_rag
from trulens_eval import TruCustomApp
from trulens_eval.ux.styles import CATEGORY

generator = StreamGenerator()

def page_setup(title, wide_mode=False, collapse_sidebar=False, visibility="public"):
    if st.get_option("client.showSidebarNavigation") and "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.rerun()

    # Handle access control
    if visibility in ("user", "admin") and not st.session_state.get("user_name"):
        st.switch_page("app.py")
    if visibility == "admin" and not st.session_state.get("admin_mode"):
        st.switch_page("app.py")

    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    LOGO = str(CURRENT_DIR / "logo.png")
    ICON_LOGO = str(CURRENT_DIR / "logo_small.png")

    st.set_page_config(
        page_title=f"LLM Evaluation: {title}",
        page_icon=ICON_LOGO,
        layout="wide" if wide_mode else "centered",
        initial_sidebar_state="collapsed" if collapse_sidebar else "auto",
    )

    st.logo(LOGO, link="https://www.snowflake.com", icon_image=ICON_LOGO)
    st.title(title)

    # Check for initial login via query_params
    if initial_user := st.query_params.get("user"):
        st.session_state.user_name = initial_user
        del st.query_params["user"]

    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    # Add page navigation
    with st.sidebar:
        st.header("LLM Evaluation")

        # st.write("")
        st.page_link("pages/about.py", label="About", icon=":material/info:")
        st.page_link("app.py", label="Chat", icon=":material/chat:")

        if st.session_state.get("user_name"):
            st.page_link("pages/account.py", label="My Account", icon=":material/account_circle:")

        if st.session_state.get("admin_mode"):
            st.subheader("Admin view")
            st.page_link("pages/analysis.py", label="Conversation Analysis", icon=":material/analytics:")
            st.page_link("pages/users.py", label="User Management", icon=":material/group:")

        st.write("")

        if user := st.session_state.get("user_name"):
            with st.popover("⚙️&nbsp; Settings", use_container_width=True):
                st.write(f"Logged in user: `{user}`")
                sidebar_container = st.container()
                if st.button("🔑&nbsp; Logout", use_container_width=True):
                    st.session_state.user_name = None
                    st.session_state.admin_mode = None
                    if visibility != "public":
                        st.switch_page("app.py")
                    else:
                        st.rerun()
        else:
            sidebar_container = st.container()
            if st.button("🔑&nbsp; Login", use_container_width=True):
                login()

    return sidebar_container


@st.experimental_dialog("Login")
def login():
    conv_mgr: ConversationManager = st.session_state.conversation_manager
    options = set([""])
    options.update(conv_mgr.list_users())
    existing = st.selectbox("Existing user:", options)
    if not existing:
        new_user = st.text_input("New user:")
    admin_mode = st.checkbox("Admin mode", value=True)
    if st.button("Submit"):
        st.session_state.user_name = existing or new_user
        st.session_state.admin_mode = admin_mode
        st.rerun()

def get_tru_app_id(model: str, temperature: float, top_p: float, max_new_tokens: int, use_rag: bool):
    # Args are hashed for cache lookup
    return f"app-prod-{model}{'-rag' if use_rag else ''} (temp-{temperature}-topp-{top_p}-maxtokens-{max_new_tokens})"

def configure_model(*, container, model_config: ModelConfig, key: str, full_width: bool = True):
    MODEL_KEY = f"model_{key}"
    TEMPERATURE_KEY = f"temperature_{key}"
    TOP_P_KEY = f"top_p_{key}"
    MAX_NEW_TOKENS_KEY = f"max_new_tokens_{key}"
    SYSTEM_PROMPT_KEY = f"system_prompt_{key}"
    USE_RAG_KEY = f"use_rag_{key}"

    # initialize app metadata for tracking
    metadata = {
        "model": st.session_state.get(MODEL_KEY, model_config.model),
        "temperature": st.session_state.get(TEMPERATURE_KEY, model_config.temperature),
        "top_p": st.session_state.get(TOP_P_KEY, model_config.top_p),
        "max_new_tokens": st.session_state.get(MAX_NEW_TOKENS_KEY, model_config.max_new_tokens),
        "use_rag": st.session_state.get(USE_RAG_KEY, model_config.use_rag),
    }

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = model_config.model
        st.session_state[TEMPERATURE_KEY] = model_config.temperature
        st.session_state[TOP_P_KEY] = model_config.top_p
        st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens
        st.session_state[USE_RAG_KEY] = model_config.use_rag
        metadata = {
                        "model": st.session_state[MODEL_KEY],
                        "temperature": st.session_state[TEMPERATURE_KEY],
                        "top_p": st.session_state[TOP_P_KEY],
                        "max_new_tokens": st.session_state[MAX_NEW_TOKENS_KEY],
                        "use_rag": st.session_state[USE_RAG_KEY],
                    }

    with container:
        with st.popover(
            f"Configure :blue[{st.session_state[MODEL_KEY]}]", use_container_width=full_width
        ):
            left1, right1 = st.columns(2)
            left2, right2 = st.columns(2)
            with left1:
                model_config.model = st.selectbox(
                    label="Select model:",
                    options=AVAILABLE_MODELS,
                    key=MODEL_KEY,
                )
                if model_config.model != st.session_state[MODEL_KEY]:
                    st.session_state[MODEL_KEY] = model_config.model

            with left2:
                SYSTEM_PROMPT_HELP = """
                    Add a system prompt which is added to the beginning
                    of each conversation.
                """
                model_config.system_prompt = st.text_area(
                    label="System Prompt:",
                    height=2,
                    key=SYSTEM_PROMPT_KEY,
                    help=SYSTEM_PROMPT_HELP,
                )
                if model_config.system_prompt != st.session_state[SYSTEM_PROMPT_KEY]:
                    st.session_state[SYSTEM_PROMPT_KEY] = model_config.system_prompt

            with right1:
                model_config.temperature = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    label="Temperature:",
                    key=TEMPERATURE_KEY,
                )
                if model_config.temperature != st.session_state[TEMPERATURE_KEY]:
                    st.session_state[TEMPERATURE_KEY] = model_config.temperature

            with right2:
                model_config.top_p = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    label="Top P:",
                    key=TOP_P_KEY,
                )
                if model_config.top_p != st.session_state[TOP_P_KEY]:
                    st.session_state[TOP_P_KEY] = model_config.top_p

                model_config.max_new_tokens = st.slider(
                    min_value=100,
                    max_value=1500,
                    step=100,
                    label="Max new tokens:",
                    key=MAX_NEW_TOKENS_KEY,
                )
                if model_config.max_new_tokens != st.session_state[MAX_NEW_TOKENS_KEY]:
                    st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens
                
                model_config.use_rag = st.toggle(
                    label="Access to Streamlit Docs",
                    value=True,
                    key=USE_RAG_KEY
                )
                if model_config.use_rag != st.session_state[USE_RAG_KEY]:
                    st.session_state[USE_RAG_KEY] = model_config.use_rag

    app_id = get_tru_app_id(**metadata)
    feedbacks = feedbacks_rag if model_config.use_rag else feedbacks_no_rag
    app = TruCustomApp(generator, app_id=app_id, metadata=metadata, feedbacks=feedbacks)
    model_config.trulens_recorder = app
    return model_config

def chat_response(
    conversation: Conversation,
    container=None,
):
    conversation.add_message(
        Message(role="assistant", content=""),
        render=False,
    )
    try:
        config = conversation.model_config
        recorder = config.trulens_recorder

        if container:
            chat = container.chat_message("assistant")
        else:
            chat = st.chat_message("assistant")
        with recorder:
            user_message, prompt = generator.prepare_prompt(conversation)
            if conversation.model_config.use_rag:
                text_response: str = generator.retrieve_and_generate_response(user_message, prompt, conversation, chat)
            else:
                text_response: str = generator.generate_response(user_message, prompt, conversation, chat)

        message = conversation.messages[-1]
        message.content = str(text_response).strip()

        if config.use_rag and recorder is not None:
            recorder.wait_for_feedback_results()
            df_results, feedback_cols = recorder.db.get_records_and_feedback([recorder.app_id])

            # Get results for most recent row
            row = df_results.iloc[-1]
            feedback_with_valid_results = sorted(
                list(filter(lambda fcol: row[fcol] != None, feedback_cols))
            )

            feedback_directions = {
                (
                    row.feedback_json.get("supplied_name", "") or
                    row.feedback_json["implementation"]["name"]
                ): (
                    "HIGHER_IS_BETTER" if row.feedback_json.get("higher_is_better", True)
                    else "LOWER_IS_BETTER"
                ) for _, row in recorder.db.get_feedback_defs().iterrows()
            }
            default_direction="HIGHER_IS_BETTER"

            def get_icon(feedback_name):
                cat = CATEGORY.of_score(
                    row[feedback_name],
                    higher_is_better=feedback_directions.get(
                        feedback_name, default_direction
                    ) == default_direction
                )
                return cat.icon

            for fcol in feedback_with_valid_results:
                message.feedbacks[fcol] = FeedbackDisplay(score=row[fcol], calls = row[f"{fcol}_calls"], icon=get_icon(feedback_name=fcol))

            # Hacky - hardcodes the call sources
            if 'Groundedness_calls' in row:
                for call in row['Groundedness_calls']:
                    message.sources.add(call['args']['source'])
            
            if 'Context Relevance_calls' in row:
                for call in row['Context Relevance_calls']:
                    message.sources.add(call['args']['context'])


    except Exception as e:
        conversation.has_error = True
        print("Error while generating chat response: " + str(e))


def generate_title(
    user_input: str,
    response_dict: Dict,
):
    SYSTEM_PROMPT = """
        You are a helpful assistant generating a brief summary title of a
        conversation based on the users input. The summary title should
        be no more than 4-5 words, with 2-3 words as a typical response.
        In general, brief is better when the title is a clear summary.

        Input will be provided in JSON format and you should specify the
        output in JSON format. Do not add any commentary or discussion.
        ONLY return the JSON.

        Here are a few examples:
        INPUT: {"input": "Hey, I'm looking for tips on planning a trip to Chicago. What should I do while I'm there?"}
        OUTPUT: {"summary": "Visiting Chicago"}

        INPUT: {"input": "I've been scripting and doing simple database work for a few years and I want to learn frontend web development. Where should I start?"}
        OUTPUT: {"summary": "Learning frontend development"}

        INPUT: {"input": "Can you share a few jokes?"}
        OUTPUT: {"summary": "Sharing jokes"}

        Ok, now your turn. Remember to only respond with the JSON.
        ------------------------------------------
    """
    conversation = Conversation()
    conversation.model_config = ModelConfig(system_prompt=SYSTEM_PROMPT)
    input_msg = json.dumps({"input": user_input})
    conversation.add_message(Message(role="user", content=input_msg), render=False)
    title_json = ""
    try:
        last_user_message, prompt = generator.prepare_prompt(conversation)
        title_json: str = generator.generate_response(last_user_message, prompt, conversation)
        
        result = json.loads(title_json)
        response_dict["output"] = result["summary"]

    except Exception as e:
        response_dict["error"] = True
        print("Error while generating title: " + str(e))
        print("Response:" + title_json)


def st_thread(target, args) -> threading.Thread:
    """Return a function as a Streamlit-safe thread"""

    thread = threading.Thread(target=target, args=args)
    add_script_run_ctx(thread, get_script_run_ctx())
    return thread

