import json
import pathlib
import threading
from typing import Dict

from conversation_manager import ConversationManager
# feedback functions
from feedback import get_feedbacks
from llm import PROVIDER_MODELS
from llm import StreamGenerator
from retrieve import AVAILABLE_RETRIEVERS
from feedback import AVAILABLE_PROVIDERS
from schema import Conversation
from schema import FeedbackDisplay
from schema import Message
from schema import ModelConfig
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import \
    get_script_run_ctx

from trulens_eval.schema.feedback import FeedbackCall
from trulens_eval.schema.feedback import FeedbackDefinition
from trulens_eval.schema.feedback import FeedbackResult
from trulens_eval.schema.record import Record
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.utils.python import Future
from trulens_eval.ux.styles import CATEGORY

generator = StreamGenerator()


def page_setup(
    title, wide_mode=False, collapse_sidebar=False, visibility="public"
):
    if st.get_option("client.showSidebarNavigation"
                    ) and "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.rerun()

    # Handle access control
    if visibility in ("user",
                      "admin") and not st.session_state.get("user_name"):
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
            st.page_link(
                "pages/account.py",
                label="My Account",
                icon=":material/account_circle:"
            )

        if st.session_state.get("admin_mode"):
            st.subheader("Admin view")
            st.page_link(
                "pages/analysis.py",
                label="Conversation Analysis",
                icon=":material/analytics:"
            )
            st.page_link(
                "pages/users.py",
                label="User Management",
                icon=":material/group:"
            )

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


def get_tru_app_id(
    model: str, temperature: float, top_p: float, max_new_tokens: int,
    use_rag: bool, retriever: str, retrieval_filter: float, provider: str, **kwargs
) -> str:
    # Args are hashed for cache'(' lookup
    return f"app-prod-{model}{'-' + retriever if use_rag else ''}{f'-retrieval-filter-' + str(retrieval_filter) if use_rag else ''} (provider-{provider}temp-{temperature}-topp-{top_p}-maxtokens-{max_new_tokens})"


def configure_model(
    *, container, model_config: ModelConfig, key: str, full_width: bool = True
):
    MODEL_KEY = f"model_{key}"
    TEMPERATURE_KEY = f"temperature_{key}"
    TOP_P_KEY = f"top_p_{key}"
    MAX_NEW_TOKENS_KEY = f"max_new_tokens_{key}"
    SYSTEM_PROMPT_KEY = f"system_prompt_{key}"
    USE_RAG_KEY = f"use_rag_{key}"
    RETRIEVAL_FILTER_KEY = f"retrieval_filter_{key}"
    RETRIEVER_KEY = f"retriever_{key}"
    PROVIDER_KEY = f"provider_{key}"

    # initialize app metadata for tracking
    metadata = {
        "model":
            st.session_state.get(MODEL_KEY, model_config.model),
        "temperature":
            st.session_state.get(TEMPERATURE_KEY, model_config.temperature),
        "top_p":
            st.session_state.get(TOP_P_KEY, model_config.top_p),
        "max_new_tokens":
            st.session_state.get(
                MAX_NEW_TOKENS_KEY, model_config.max_new_tokens
            ),
        "use_rag":
            st.session_state.get(USE_RAG_KEY, model_config.use_rag),
        "retriever":
            st.session_state.get(RETRIEVER_KEY, model_config.retriever),
        "retrieval_filter":
            st.session_state.get(RETRIEVAL_FILTER_KEY, model_config.retrieval_filter),
        "provider":
            st.session_state.get(PROVIDER_KEY, model_config.provider),
    }

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = model_config.model
        st.session_state[TEMPERATURE_KEY] = model_config.temperature
        st.session_state[TOP_P_KEY] = model_config.top_p
        st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens
        st.session_state[USE_RAG_KEY] = model_config.use_rag
        st.session_state[RETRIEVER_KEY] = model_config.retriever
        st.session_state[PROVIDER_KEY] = model_config.provider
        metadata = {
            "model": st.session_state[MODEL_KEY],
            "temperature": st.session_state[TEMPERATURE_KEY],
            "top_p": st.session_state[TOP_P_KEY],
            "max_new_tokens": st.session_state[MAX_NEW_TOKENS_KEY],
            "use_rag": st.session_state[USE_RAG_KEY],
            "retriever": st.session_state[RETRIEVER_KEY],
            "retrieval_filter": st.session_state.get(RETRIEVAL_FILTER_KEY),
            "provider": st.session_state[PROVIDER_KEY],
        }

    with container:
        with st.popover(f"Configure :blue[{st.session_state[MODEL_KEY]}]",
                        use_container_width=full_width):
            left1, right1 = st.columns(2)
            left2, right2 = st.columns(2)
            with left1:
                model_config.provider = st.selectbox(
                    label="Select provider:",
                    options=AVAILABLE_PROVIDERS,
                    key=PROVIDER_KEY,
                )
                if model_config.provider != st.session_state[PROVIDER_KEY]:
                    st.session_state[PROVIDER_KEY] = model_config.provider

                model_config.model = st.selectbox(
                    label="Select model:",
                    options=PROVIDER_MODELS[model_config.provider].keys(),
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
                if model_config.system_prompt != st.session_state[
                        SYSTEM_PROMPT_KEY]:
                    st.session_state[SYSTEM_PROMPT_KEY
                                    ] = model_config.system_prompt

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
                if model_config.max_new_tokens != st.session_state[
                        MAX_NEW_TOKENS_KEY]:
                    st.session_state[MAX_NEW_TOKENS_KEY
                                    ] = model_config.max_new_tokens

                model_config.use_rag = st.toggle(
                    label="Access to Streamlit Docs",
                    value=True,
                    key=USE_RAG_KEY
                )
                if model_config.use_rag != st.session_state[USE_RAG_KEY]:
                    st.session_state[USE_RAG_KEY] = model_config.use_rag
            model_config.retriever = st.selectbox(
                label="Select retriever:",
                options=AVAILABLE_RETRIEVERS.keys(),
                key=RETRIEVER_KEY,
            )
            if model_config.retriever != st.session_state[RETRIEVER_KEY]:
                st.session_state[RETRIEVER_KEY] = model_config.retriever

                model_config.retrieval_filter = st.slider(
                    min_value=0,
                    max_value=1,
                    step=0.1,
                    label="Context Relevance Filter for Retrieval",
                    key=RETRIEVAL_FILTER_KEY
                )

                if model_config.retrieval_filter != st.session_state[RETRIEVAL_FILTER_KEY]:
                    st.session_state[RETRIEVAL_FILTER_KEY] = model_config.retrieval_filter

    app_id = get_tru_app_id(**metadata)
    feedbacks = get_feedbacks(model_config.provider, model_config.use_rag)
    app = TruCustomApp(
        generator, app_id=app_id, metadata=metadata, feedbacks=feedbacks
    )
    model_config.trulens_recorder = app
    return model_config


def get_icon(fdef: FeedbackDefinition, result: float):
    cat = CATEGORY.of_score(
        result or 0,
        higher_is_better=fdef.higher_is_better
        if fdef.higher_is_better is not None else True
    )
    return cat.icon


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
        with recorder as context:
            user_message, prompt = generator.prepare_prompt(conversation)
            if conversation.model_config.use_rag:
                text_response: str = generator.retrieve_and_generate_response(
                    user_message, prompt, conversation, chat
                )
            else:
                text_response: str = generator.generate_response(
                    user_message, prompt, conversation, chat
                )

        record: Record = context.get()

        message = conversation.messages[-1]
        message.content = str(text_response).strip()

        # Check if we have to return feedback function results for the RAG triad.
        if not config.use_rag or recorder is None:
            return

        # If no feedback functions are returning we can skip it.
        if record.feedback_and_future_results is None:
            return

        # Let this be updated async and streamlit pick it up later
        def update_result(
            fdef: FeedbackDefinition, fres: Future[FeedbackResult]
        ):
            result = fres.result()
            calls = result.calls
            score = result.result or 0

            message.feedbacks[fdef.name] = FeedbackDisplay(
                score=score, calls=calls, icon=get_icon(fdef, score)
            )

            # Hacky - hardcodes behavior based on feedback name
            if fdef.name == "Groundedness":
                for call in calls:
                    message.sources.add(call.args['source'])

            if fdef.name == "Context Relevance":
                for call in calls:
                    message.sources.add(call.args['context'])

        for feedback_def, future_result in record.feedback_and_future_results:
            t = st_thread(
                target=update_result, args=(feedback_def, future_result)
            )
            t.start()

    except Exception as e:
        conversation.has_error = True
        print("Error while generating chat response: " + str(e))


def generate_title(
    user_input: str,
    response_dict: Dict,
    model_config: ModelConfig
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
    conversation.model_config = ModelConfig(
        system_prompt=SYSTEM_PROMPT, 
        provider=model_config.provider, 
        model=model_config.model
    )
    input_msg = json.dumps({"input": user_input})
    conversation.add_message(
        Message(role="user", content=input_msg), render=False
    )
    title_json = ""
    try:
        last_user_message, prompt = generator.prepare_prompt(conversation)
        title_json: str = generator.generate_response(
            last_user_message, prompt, conversation, should_write=False
        )
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
