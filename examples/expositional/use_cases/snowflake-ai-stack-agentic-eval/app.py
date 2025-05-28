import streamlit as st
from trulens.core import TruSession
from trulens.apps.app import TruApp
from trulens.dashboard import streamlit as trulens_st
from src.graph import MultiAgentWorkflow
import os

st.set_page_config(page_title="Snowflake Agentic Evaluation Demo", page_icon="❄️", layout="centered", initial_sidebar_state="collapsed", menu_items=None)

st.subheader("Using the ❄️ AI Stack to build agentic workflows and evaluate them with TruLens")

# Initialize TruLens for observability (similar to notebook example)
if "tru_session" not in st.session_state:
    tru_session = TruSession()
    st.session_state.tru_session = tru_session
    # Reset database on app startup
    st.session_state.tru_session.reset_database()

# Initialize session state if not already set
if "muti_agent_workflow" not in st.session_state:
    st.session_state.muti_agent_workflow = None
if "tru_agentic_eval_app" not in st.session_state:
    st.session_state.tru_agentic_eval_app = None
if "messages" not in st.session_state:
    st.session_state.messages = []



# Create the TruAgent instance only once
if st.session_state.muti_agent_workflow is None:
    st.session_state.muti_agent_workflow = MultiAgentWorkflow(
        search_max_results=int(os.environ.get("SEARCH_MAX_RESULTS", "5")),
        llm_model=os.environ.get("LLM_MODEL_NAME", "gpt-4o"),
        reasoning_model=os.environ.get("REASONING_MODEL_NAME", "o1"),
    )

    st.session_state.tru_agentic_eval_app = TruApp(
        st.session_state.muti_agent_workflow,
        app_name="Langgraph Agentic Evaluation",
        app_version="trajectory-eval-oss",
        # TODO main method?
    )
    st.success("Langgraph workflow compiled!")

st.markdown("---")

# Render existing conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat interface using st.chat_input and st.chat_message
user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    if st.session_state.tru_agentic_eval_app is None:
        st.error("Please build the multi-agent workflow graph first.")
    else:
        message_container = st.chat_message("assistant")
        message_area = message_container.empty()
        full_response = ""
        with st.session_state.tru_agentic_eval_app as recording:
                # TODO: messages for chat history not used in the agent graph for now
            events = st.session_state.muti_agent_workflow.invoke_agent_graph(user_input)

            for event in events:
                full_response += list(event.values())[0]["messages"][0].content
                message_area.markdown(full_response)

        st.session_state.tru_session.force_flush()
        record_id = recording.get()

        st.session_state.tru_session.wait_for_record(record_id)
        trulens_st.trulens_trace(record=record_id)

        # Add the assistant response to session state - only once!
        st.session_state.messages.append({"role": "assistant", "content": full_response})
