import streamlit as st
from src.retrieval import VectorStore
from src.generation import ChatModel
from src.rag import Rag

# Add imports for observability
from src.observability import start_observability, create_evals
from trulens.providers.openai import OpenAI
from trulens.apps.app import TruApp
from trulens.dashboard import streamlit as trulens_st
import os

st.set_page_config(
    page_title="Snowflake AI Stack",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.subheader("Using the ❄️ AI Stack")

# Initialize TruLens for observability (similar to notebook example)
if "tru_session" not in st.session_state:
    st.session_state.tru_session = start_observability()
    # Reset database on app startup
    st.session_state.tru_session.reset_database()

# Initialize session state if not already set
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "tru_rag" not in st.session_state:
    st.session_state.tru_rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def setup_vector_store():
    vector_store = VectorStore()
    docs = vector_store.load_text_files(file_path="./data.txt")
    chunks = vector_store.split_documents(documents=docs)
    vector_store.add_chunks(chunks)
    return vector_store


if st.session_state.vector_store is None:
    st.session_state.vector_store = setup_vector_store()

# Create the RAG instance only once
if st.session_state.vector_store is not None and st.session_state.rag is None:
    chat_model = ChatModel(
        generation_model_name=os.environ.get("GENERATION_MODEL_NAME")
    )
    st.session_state.rag = Rag(
        chat_model=chat_model,
        vector_store=st.session_state.vector_store,
        use_context_filter=os.environ.get("USE_CONTEXT_FILTER"),
    )

    # Set up TruLens observability
    provider = OpenAI(
        model_engine=os.environ.get("GENERATION_MODEL_NAME"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    evals = create_evals(provider=provider)
    st.session_state.tru_rag = TruApp(
        st.session_state.rag,
        app_name="RAG",
        app_version="snowflake-oss",
        feedbacks=evals,
    )
    st.success("Knowledge Base Loaded!")

st.markdown("---")

# Render existing conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat interface using st.chat_input and st.chat_message
user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    if st.session_state.rag is None:
        st.error("Please load a knowledge base first.")
    else:
        message_container = st.chat_message("assistant")
        message_area = message_container.empty()
        full_response = ""  # Initialize to collect the full response

        # Use TruLens to track the RAG application with streaming enabled
        with st.session_state.tru_rag as recording:
            with st.spinner("Thinking..."):
                # full_response = st.session_state.rag.retrieve_and_generate(user_input, st.session_state.messages)
                # message_area.markdown(full_response)
                generator = st.session_state.rag.retrieve_and_generate_stream(
                    user_input, st.session_state.messages
                )
                for chunk in generator:
                    if chunk is not None:
                        full_response += chunk
                        message_area.markdown(full_response)

        st.session_state.tru_session.force_flush()
        record = recording.get()
        trulens_st.trulens_trace(record=record.record_id)

        # Add the assistant response to session state - only once!
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
