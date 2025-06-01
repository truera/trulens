import streamlit as st
from trulens.core import TruSession
from trulens.apps.app import TruApp
from trulens.dashboard import streamlit as trulens_st
from src.graph import MultiAgentWorkflow
from src.agentic_evals import CustomTrajEval, create_traj_eval

import os
import json
import re
import glob

st.set_page_config(page_title="Trustworthy Deep Research Agent", page_icon="❄️", layout="centered", initial_sidebar_state="collapsed", menu_items=None)

st.subheader("❄️ Trustworthy Deep Research Agent")

# Initialize TruLens for observability (similar to notebook example)
if "tru_session" not in st.session_state:
    tru_session = TruSession()
    st.session_state.tru_session = tru_session
    # Reset database on app startup
    st.session_state.tru_session.reset_database()

# Initialize session state if not already set
if "multi_agent_workflow" not in st.session_state:
    st.session_state.multi_agent_workflow = None
if "tru_agentic_eval_app" not in st.session_state:
    st.session_state.tru_agentic_eval_app = None
if "messages" not in st.session_state:
    st.session_state.messages = []



# Create the TruAgent instance only once
if st.session_state.multi_agent_workflow is None:
    st.session_state.multi_agent_workflow = MultiAgentWorkflow(
        search_max_results=int(os.environ.get("SEARCH_MAX_RESULTS", "5")),
        llm_model=os.environ.get("LLM_MODEL_NAME", "gpt-4o"),
        reasoning_model=os.environ.get("REASONING_MODEL_NAME", "o1"),
    )
    traj_provider = CustomTrajEval(model_engine=os.environ.get("LLM_MODEL_NAME", "gpt-4o")) # note: reasoning model is not yet supported in TruLens OpenAI provider

    f_traj_eval = create_traj_eval(provider=traj_provider)

    st.session_state.tru_agentic_eval_app = TruApp(
        st.session_state.multi_agent_workflow,
        app_name="Langgraph Agentic Evaluation",
        app_version="trajectory-eval-oss",
        feedbacks=[f_traj_eval]
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
        full_response = ""
        with st.session_state.tru_agentic_eval_app as recording:
                # TODO: messages for chat history not used in the agent graph for now
            events = st.session_state.multi_agent_workflow.invoke_agent_graph(user_input)

            for event in events:
                if not event:
                    continue
                node_name = list(event.keys())[0]
                update = event[node_name]
                messages = update.get("messages", [])
                if not messages:
                    continue
                last_msg = messages[-1]
                content = last_msg.content
                role = getattr(last_msg, "name", node_name)  # fallback to node_name

                if node_name == "orchestrator":
                    # Try to parse the content as JSON to extract 'goto' and 'reason'
                    try:
                        parsed = json.loads(content)
                        goto = parsed.get("goto", "[missing]")
                        reason = parsed.get("reason", "[missing]")
                        st.chat_message("orchestrator", avatar = "🧑‍💼").markdown(f"**Orchestrator:**\n- **Next node:** `{goto}`\n- **Reason:** {reason}")
                    except Exception:
                        st.chat_message("orchestrator", avatar = "🧑‍💼").markdown(f"**Orchestrator:** \n{content}")
                elif node_name == "researcher":
                    with st.expander("🔬 Researcher Output", expanded=False):
                        st.write(f"\n {content}")
                elif node_name == "chart_generator":
                    # Always show the text output
                    st.chat_message("chart_generator", avatar = "📊").write(f"**Chart Generator:** \n{content}")
                    # Find and display the most recent .png file in images/
                    images_dir = "images"
                    png_files = glob.glob(os.path.join(images_dir, "*.png"))
                    if png_files:
                        latest_png = max(png_files, key=os.path.getmtime)
                        try:
                            with open(latest_png, "rb") as img_file:
                                img_bytes = img_file.read()
                            st.image(img_bytes)
                        except Exception as e:
                            st.warning(f"Could not display chart image: {e}")
                    else:
                        st.warning("No chart image found to display.")
                elif node_name.endswith("_eval"):
                    # Try to extract the score and eval name for the expander title
                    eval_name = role.replace("_", " ").title() if role else node_name.replace("_", " ").title()
                    # For research_eval and chart_eval, split into individual metrics
                    if node_name in ["research_eval", "chart_eval"]:
                        # Split by lines and group by metric
                        lines = content.split("\n")
                        current_metric = None
                        metric_lines = {}
                        for line in lines:
                            metric_match = re.match(r"-?\s*([A-Za-z ]+): ([0-9.]+/3) — (.*)", line)
                            if metric_match:
                                current_metric = metric_match.group(1).strip()
                                score = metric_match.group(2).strip()
                                reason = metric_match.group(3).strip()
                                metric_lines[current_metric] = {"score": score, "reason": reason}
                            elif current_metric and line.strip():
                                # Append additional lines to the reason
                                metric_lines[current_metric]["reason"] += "\n" + line.strip()
                        # Render each metric in its own expander
                        for metric, data in metric_lines.items():
                            expander_title = f"**{metric} ({data['score']})**"
                            with st.expander(expander_title, expanded=False):
                                st.markdown(data["reason"])
                    else:
                        # For other evals, use previous logic
                        score_match = re.search(r"(\d+(?:\.\d+)?/3)", content)
                        score_str = score_match.group(1) if score_match else ""
                        expander_title = f"**{eval_name} ({score_str})**" if score_str else eval_name
                        with st.expander(expander_title, expanded=False):
                            st.markdown(content)
                else:
                    st.chat_message("assistant").write(content)

        st.session_state.tru_session.force_flush()
        record_id = recording.get()

        st.session_state.tru_session.wait_for_record(record_id)
        trulens_st.trulens_trace(record=record_id)
        trulens_st.trulens_feedback(record_id=record_id)

        # Add the assistant response to session state - only once!
        st.session_state.messages.append({"role": "assistant", "content": full_response})
