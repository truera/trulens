"""
Streamlit Application for Cortex Agent Prompt Optimization

A user-friendly interface for:
1. Selecting and managing Cortex Agents
2. Loading evaluation datasets
3. Running agents and collecting responses
4. Gathering human feedback (optional)
5. Running LLM judge evaluations
6. Optimizing agent prompts
7. Comparing and applying results
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from cortex_agent_manager import AgentInstructions
from cortex_agent_manager import CortexAgentManager
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import requests
from snowflake.snowpark import Session
import streamlit as st
from trulens.otel.semconv.trace import SpanAttributes

# Optional imports for evaluation
try:
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession
    from trulens.core.feedback.selector import Trace
    from trulens.providers.openai.provider import OpenAI as fOpenAI

    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    st.warning("TruLens not available. Evaluation features will be limited.")


# ==================== Configuration ====================

load_dotenv()

# LLM for prompt generation
LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7

# Page config
st.set_page_config(
    page_title="Cortex Agent Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== Session State Initialization ====================


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Configuration
        "account_url": os.getenv("SNOWFLAKE_ACCOUNT_URL", ""),
        "database": os.getenv("SNOWFLAKE_DATABASE", ""),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", ""),
        "pat": os.getenv("SNOWFLAKE_PAT", ""),
        "agent_manager": None,
        "tru_session": None,
        "tru_provider": None,
        # Agent selection
        "available_agents": [],
        "selected_agent": None,
        "agent_details": None,
        # Data
        "eval_dataset": None,  # DataFrame with columns: question, ground_truth (optional)
        # Execution results
        "agent_responses": None,  # DataFrame with: question, agent_response, request_id, timestamp
        # Feedback
        "feedback_data": {},  # Dict[row_idx, feedback_dict]
        # Evaluations
        "evaluation_results": None,  # DataFrame with metrics
        # Optimization
        "optimization_history": [],  # List of dicts with iteration data
        "optimization_running": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== Helper Functions ====================


def get_agent_manager() -> Optional[CortexAgentManager]:
    """Get or create CortexAgentManager from session state."""
    if st.session_state.agent_manager is None:
        try:
            st.session_state.agent_manager = CortexAgentManager(
                account_url=st.session_state.account_url,
                auth_token=st.session_state.pat,
                database=st.session_state.database,
                schema=st.session_state.schema,
            )
        except Exception as e:
            st.error(f"Failed to initialize Agent Manager: {e}")
            return None
    return st.session_state.agent_manager


def get_trulens_components() -> Tuple[Optional[TruSession], Optional[Any]]:
    """Get or create TruLens components from session state."""
    if not TRULENS_AVAILABLE:
        return None, None

    if (
        st.session_state.tru_session is None
        or st.session_state.tru_provider is None
    ):
        try:
            connection_params = {
                "account": st.session_state.account_url.replace("https://", "")
                .replace(".snowflakecomputing.com", "")
                .rstrip("/"),
                "user": os.getenv("SNOWFLAKE_USER"),
                "password": st.session_state.pat,
                "database": st.session_state.database,
                "schema": st.session_state.schema,
            }
            snowpark_session = Session.builder.configs(
                connection_params
            ).create()
            st.session_state.tru_session = TruSession(
                SnowflakeConnector(snowpark_session=snowpark_session)
            )
            st.session_state.tru_provider = fOpenAI(model_engine=LLM_MODEL)
        except Exception as e:
            st.error(f"Failed to initialize TruLens: {e}")
            return None, None

    return st.session_state.tru_session, st.session_state.tru_provider


def extract_agent_response_text(response: dict) -> str:
    """Extract text from agent response."""
    message_obj = response.get("message", {})
    content = message_obj.get("content", [])

    for content_item in content:
        if content_item.get("type") == "text":
            return content_item.get("text", "")
    return ""


def parse_agent_instructions(agent_details: Optional[Dict]) -> Dict[str, str]:
    """
    Parse agent instructions from agent details response.

    The Snowflake API returns instructions inside a JSON string in the agent_spec field.
    This helper extracts and parses that structure.

    Returns:
        Dict with 'system', 'response', 'orchestration' keys
    """
    if not agent_details:
        return {}

    try:
        agent_spec = agent_details.get("agent_spec", "{}")
        if isinstance(agent_spec, str):
            agent_spec_obj = json.loads(agent_spec)
        else:
            agent_spec_obj = agent_spec

        return agent_spec_obj.get("instructions", {})
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def submit_feedback_to_api(
    agent_manager: CortexAgentManager,
    agent_name: str,
    request_id: str,
    positive: bool,
    feedback_message: str,
    categories: List[str],
    thread_id: Optional[int] = None,
) -> bool:
    """Submit feedback via REST API."""
    url = f"{agent_manager.base_api_url}/databases/{agent_manager.database}/schemas/{agent_manager.schema}/agents/{agent_name}:feedback"

    body = {
        "request_id": request_id,
        "positive": positive,
        "feedback_message": feedback_message,
        "categories": categories,
    }

    if thread_id is not None:
        body["thread_id"] = thread_id

    try:
        response = requests.post(url, json=body, headers=agent_manager.headers)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to submit feedback: {e}")
        return False


def propose_prompt_variant(
    role: str,
    previous_instruction: Optional[str] = None,
    eval_feedback: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate a new prompt variant using LLM."""
    client = OpenAI()

    base_prompt = (
        f"Generate a new {role} for an AI agent. Keep it concise and effective."
    )

    if previous_instruction:
        base_prompt += f"\n\nUse the previous {role} as a reference:\n```\n{previous_instruction}\n```"

    if eval_feedback:
        base_prompt += f"\n\nEvaluation feedback from previous version:\n```\n{eval_feedback}\n```"

    base_prompt += f"""

The new {role} should be different and improved. Focus on:
- Clarity and specificity
- Factual accuracy
- Usefulness and relevance
- Conciseness
"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert prompt engineer.",
                },
                {"role": "user", "content": base_prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Failed to generate prompt variant: {e}")
        return previous_instruction or ""


def evaluate_response(
    question: str,
    agent_response_dict: dict,
    ground_truth: Optional[str],
    request_id: str,
    agent_name: str,
    tru_session: Optional[TruSession],
    tru_provider: Optional[Any],
) -> Dict[str, Any]:
    """
    Evaluate agent response with available metrics.

    Returns dict with metric_name -> {"score": float, "reason": str}
    """
    metrics = {}
    agent_response_text = extract_agent_response_text(agent_response_dict)

    # Ground-truth based evaluation (if available)
    if ground_truth and tru_provider:
        try:
            accuracy_prompt = f"""
You are an impartial evaluator.

Question: {question}
Agent Response: {agent_response_text}
Ground Truth: {ground_truth}

Score the agent response on its accuracy given the question and ground truth. Numerical score must be between 0 and 1.

Criteria: <Provide the criteria for this evaluation>
Supporting Evidence: <Provide your reasons step by step>
Score: <The score based on the criteria>
"""
            accuracy_score, accuracy_reason = (
                tru_provider.generate_score_and_reasons(
                    system_prompt=accuracy_prompt,
                    temperature=0.0,
                    min_score_val=0,
                    max_score_val=1,
                )
            )
            metrics["accuracy"] = {
                "score": accuracy_score,
                "reason": accuracy_reason.get("reason", ""),
            }
        except Exception as e:
            st.warning(f"Failed to evaluate accuracy: {e}")

    # GPA evaluations (requires trace from event table)
    if tru_session and tru_provider and request_id and agent_name:
        try:
            events_df = tru_session.get_events(
                app_name=agent_name,
                app_version=None,
                record_ids=[request_id],
            )

            if not events_df.empty:
                trace = Trace()
                trace.events = events_df

                gpa_evaluators = {
                    "logical_consistency": tru_provider.logical_consistency_with_cot_reasons,
                    "execution_efficiency": tru_provider.execution_efficiency_with_cot_reasons,
                    "plan_adherence": tru_provider.plan_adherence_with_cot_reasons,
                    "plan_quality": tru_provider.plan_quality_with_cot_reasons,
                }

                for metric_name, eval_func in gpa_evaluators.items():
                    try:
                        score, reason_dict = eval_func(trace=trace)
                        metrics[metric_name] = {
                            "score": score,
                            "reason": reason_dict.get("reason", ""),
                        }
                    except Exception as e:
                        st.warning(f"Failed to evaluate {metric_name}: {e}")
        except Exception as e:
            st.warning(f"Failed to retrieve trace for GPA evaluation: {e}")

    return metrics


# ==================== UI Sections ====================


def render_sidebar():
    """Render sidebar with configuration."""
    with st.sidebar:
        st.title("🎯 Agent Optimizer")
        st.markdown("---")

        st.subheader("Configuration")

        st.session_state.account_url = st.text_input(
            "Snowflake Account URL",
            value=st.session_state.account_url,
            help="e.g., https://myaccount.snowflakecomputing.com",
        )

        st.session_state.database = st.text_input(
            "Database",
            value=st.session_state.database,
        )

        st.session_state.schema = st.text_input(
            "Schema",
            value=st.session_state.schema,
        )

        st.session_state.pat = st.text_input(
            "PAT (Personal Access Token)",
            value=st.session_state.pat,
            type="password",
        )

        st.markdown("---")

        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["account_url", "database", "schema", "pat"]:
                    del st.session_state[key]
            st.rerun()

        # Debug section
        with st.expander("🐛 Debug Info", expanded=False):
            st.markdown("**Session State Status:**")
            st.write(
                f"- Agent Manager: {'✅ Connected' if st.session_state.agent_manager else '❌ Not connected'}"
            )
            st.write(
                f"- Selected Agent: {st.session_state.selected_agent or 'None'}"
            )
            st.write(
                f"- Dataset Loaded: {'✅ Yes' if st.session_state.eval_dataset is not None else '❌ No'}"
            )
            if st.session_state.eval_dataset is not None:
                st.write(f"  - Rows: {len(st.session_state.eval_dataset)}")
            st.write(
                f"- Agent Responses: {'✅ Yes' if st.session_state.agent_responses is not None else '❌ No'}"
            )
            if st.session_state.agent_responses is not None:
                st.write(f"  - Rows: {len(st.session_state.agent_responses)}")
            st.write(
                f"- Evaluations: {'✅ Yes' if st.session_state.evaluation_results is not None else '❌ No'}"
            )
            st.write(
                f"- Optimization History: {len(st.session_state.optimization_history)} runs"
            )
            st.write(
                f"- **Optimization Running: {'🔴 TRUE (BLOCKED!)' if st.session_state.optimization_running else '🟢 False'}**"
            )

            if st.session_state.optimization_running:
                st.warning(
                    "⚠️ Optimization is marked as running! This might be stuck."
                )
                if st.button(
                    "🔓 Force Reset Optimization Flag", key="force_reset"
                ):
                    st.session_state.optimization_running = False
                    st.success("Reset! Please try again.")
                    st.rerun()


def render_agent_selection():
    """Section 1: Agent Selection."""
    st.header("1️⃣ Agent Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔍 Load Agents", use_container_width=True):
            agent_manager = get_agent_manager()
            if agent_manager:
                try:
                    with st.spinner("Loading agents..."):
                        agents = agent_manager.list_agents()
                        st.session_state.available_agents = agents
                        st.success(f"Found {len(agents)} agents")
                except Exception as e:
                    st.error(f"Failed to load agents: {e}")

    if st.session_state.available_agents:
        agent_names = [
            a.get("name", "Unknown") for a in st.session_state.available_agents
        ]

        selected = st.selectbox(
            "Select Agent",
            options=agent_names,
            index=agent_names.index(st.session_state.selected_agent)
            if st.session_state.selected_agent in agent_names
            else 0,
        )

        if selected != st.session_state.selected_agent:
            st.session_state.selected_agent = selected
            st.session_state.agent_details = None

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📋 Load Agent Details", use_container_width=True):
                agent_manager = get_agent_manager()
                if agent_manager and st.session_state.selected_agent:
                    try:
                        with st.spinner("Loading agent details..."):
                            details = agent_manager.describe_agent(
                                st.session_state.selected_agent
                            )
                            st.session_state.agent_details = details
                            st.success(
                                f"✅ Loaded details for {st.session_state.selected_agent}"
                            )
                    except Exception as e:
                        st.error(f"Failed to load agent details: {e}")

        with col2:
            if st.button(
                "🔄",
                use_container_width=True,
                help="Force refresh agent details",
            ):
                st.session_state.agent_details = None
                st.rerun()

        if st.session_state.agent_details:
            with st.expander("📋 Current Agent Instructions", expanded=True):
                instructions = parse_agent_instructions(
                    st.session_state.agent_details
                )

                system = instructions.get("system", "")
                response = instructions.get("response", "")
                orchestration = instructions.get("orchestration", "")

                st.markdown(f"**System:** {system if system else '*(empty)*'}")
                st.markdown(
                    f"**Response:** {response if response else '*(empty)*'}"
                )
                st.markdown(
                    f"**Orchestration:** {orchestration if orchestration else '*(empty)*'}"
                )

            # Debug: Show raw response structure (outside the expander to avoid nesting)
            with st.expander("🔍 Raw API Response (Debug)", expanded=False):
                st.json(st.session_state.agent_details)
    else:
        st.info("👆 Click 'Load Agents' to get started")


def render_data_loading():
    """Section 2: Data Loading."""
    st.header("2️⃣ Data Loading")

    tab1, tab2, tab3 = st.tabs([
        "📤 Upload CSV",
        "📊 Snowflake Table",
        "🔍 Event Table",
    ])

    with tab1:
        st.markdown(
            "Upload a CSV file with columns: `question`, `ground_truth` (optional)"
        )
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_csv_data = df
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")

        # Display and confirm outside the upload conditional
        if hasattr(st.session_state, "uploaded_csv_data"):
            st.dataframe(
                st.session_state.uploaded_csv_data.head(10),
                use_container_width=True,
            )
            if st.button("✅ Use This Dataset", key="use_csv"):
                st.session_state.eval_dataset = (
                    st.session_state.uploaded_csv_data
                )
                st.success(f"Loaded {len(st.session_state.eval_dataset)} rows")

    with tab2:
        st.markdown("Load data from a Snowflake table")
        table_name = st.text_input(
            "Table Name (e.g., MY_DATABASE.MY_SCHEMA.MY_TABLE)"
        )

        if st.button("📥 Load from Table", key="load_table"):
            if table_name:
                try:
                    # TODO: Implement table loading via Snowpark
                    st.warning(
                        "Table loading not yet implemented. Use CSV upload for now."
                    )
                except Exception as e:
                    st.error(f"Failed to load table: {e}")

    with tab3:
        st.markdown("Retrieve data from Snowflake AI Observability Event Table")

        # Note: These inputs are for future implementation
        # st.date_input("Start Date", key="event_start_date")
        # st.date_input("End Date", key="event_end_date")
        # st.text_input("Thread ID (optional)", key="event_thread_id")

        if st.button("🔍 Query Event Table", key="query_events"):
            tru_session = get_trulens_components()[0]
            events = tru_session.connector.db.get_events(
                app_name=st.session_state.selected_agent
            )
            inputs = []
            ground_truths = []
            for _, event in events.iterrows():
                record_attributes = json.loads(event["record_attributes"])
                curr_input = record_attributes.get(
                    SpanAttributes.RECORD_ROOT.INPUT
                )
                if curr_input:
                    curr_ground_truth = record_attributes.get(
                        SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT
                    )
                else:
                    curr_ground_truth = None
                if curr_input not in inputs:
                    inputs.append(curr_input)
                    ground_truths.append(curr_ground_truth)
            if ground_truths and all(ground_truths):
                df = pd.DataFrame({
                    "question": inputs,
                    "ground_truth": ground_truths,
                })
            else:
                df = pd.DataFrame({"question": inputs})
            st.session_state.queried_event_data = df

        # Display and confirm outside the query button conditional
        if hasattr(st.session_state, "queried_event_data"):
            st.dataframe(
                st.session_state.queried_event_data.head(10),
                use_container_width=True,
            )
            if st.button("✅ Use This Dataset", key="use_event_table"):
                st.session_state.eval_dataset = (
                    st.session_state.queried_event_data
                )
                st.success(f"Loaded {len(st.session_state.eval_dataset)} rows")

    # Display current dataset
    if st.session_state.eval_dataset is not None:
        st.markdown("---")
        st.subheader("Current Dataset")
        st.dataframe(st.session_state.eval_dataset, use_container_width=True)
        st.info(f"📊 {len(st.session_state.eval_dataset)} rows loaded")


def render_agent_execution():
    """Section 3: Agent Execution."""
    st.header("3️⃣ Agent Execution")

    if st.session_state.eval_dataset is None:
        st.warning("⚠️ Please load a dataset first (Section 2)")
        return

    if not st.session_state.selected_agent:
        st.warning("⚠️ Please select an agent first (Section 1)")
        return

    st.info(
        f"Ready to run agent **{st.session_state.selected_agent}** on {len(st.session_state.eval_dataset)} questions"
    )

    if st.button(
        "▶️ Run Agent on Dataset", use_container_width=True, type="primary"
    ):
        agent_manager = get_agent_manager()
        if not agent_manager:
            return

        try:
            # Create thread
            with st.spinner("Creating conversation thread..."):
                thread_id = agent_manager.create_thread(
                    origin_application="optimizer"
                )

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in st.session_state.eval_dataset.iterrows():
                status_text.text(
                    f"Processing question {idx + 1}/{len(st.session_state.eval_dataset)}"
                )

                question = row.get("question", "")
                if not question:
                    continue

                try:
                    response = agent_manager.send_message(
                        agent_name=st.session_state.selected_agent,
                        thread_id=thread_id,
                        message=question,
                    )

                    results.append({
                        "question": question,
                        "agent_response": extract_agent_response_text(response),
                        "response_dict": response,  # Store full response
                        "request_id": response.get("request_id"),
                        "thread_id": thread_id,
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception as e:
                    st.error(f"Failed on question {idx}: {e}")
                    results.append({
                        "question": question,
                        "agent_response": f"ERROR: {e}",
                        "response_dict": {},
                        "request_id": None,
                        "thread_id": thread_id,
                        "timestamp": datetime.now().isoformat(),
                    })

                progress_bar.progress(
                    (idx + 1) / len(st.session_state.eval_dataset)
                )

            st.session_state.agent_responses = pd.DataFrame(results)
            status_text.text("✅ Completed!")
            st.success(f"Processed {len(results)} questions")

        except Exception as e:
            st.error(f"Failed to run agent: {e}")

    # Display results
    if st.session_state.agent_responses is not None:
        st.markdown("---")
        st.subheader("Results")

        display_df = st.session_state.agent_responses[
            ["question", "agent_response", "request_id", "timestamp"]
        ]
        st.dataframe(display_df, use_container_width=True)


def render_feedback_collection():
    """Section 4: Human Feedback Collection (Optional)."""
    st.header("4️⃣ Human Feedback (Optional)")

    if st.session_state.agent_responses is None:
        st.warning("⚠️ Please run the agent first (Section 3)")
        return

    st.info("💡 Provide feedback on agent responses to help with optimization")

    # Pagination
    items_per_page = 5
    total_items = len(st.session_state.agent_responses)
    total_pages = (total_items + items_per_page - 1) // items_per_page

    page = (
        st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
    )
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)

    for idx in range(start_idx, end_idx):
        row = st.session_state.agent_responses.iloc[idx]

        with st.expander(f"Question {idx + 1}: {row['question'][:100]}..."):
            st.markdown("**Agent Response:**")
            st.markdown(row["agent_response"])

            st.markdown("---")
            st.markdown("**Provide Feedback:**")

            col1, col2 = st.columns([1, 3])

            with col1:
                positive = st.radio(
                    "Rating",
                    options=[True, False],
                    format_func=lambda x: "👍 Positive" if x else "👎 Negative",
                    key=f"rating_{idx}",
                )

            with col2:
                categories = st.multiselect(
                    "Categories",
                    options=[
                        "Helpful",
                        "Accurate",
                        "Clear",
                        "Complete",
                        "Incomplete",
                        "Incorrect",
                        "Confusing",
                    ],
                    key=f"categories_{idx}",
                )

            feedback_message = st.text_area(
                "Detailed Feedback",
                key=f"feedback_{idx}",
                placeholder="Optional: Provide specific feedback...",
            )

            if st.button("💾 Submit Feedback", key=f"submit_{idx}"):
                agent_manager = get_agent_manager()
                if agent_manager and row["request_id"]:
                    success = submit_feedback_to_api(
                        agent_manager=agent_manager,
                        agent_name=st.session_state.selected_agent,
                        request_id=row["request_id"],
                        positive=positive,
                        feedback_message=feedback_message,
                        categories=categories,
                        thread_id=int(row["thread_id"])
                        if row.get("thread_id")
                        else None,
                    )

                    if success:
                        # Store locally as well
                        st.session_state.feedback_data[idx] = {
                            "positive": positive,
                            "categories": categories,
                            "message": feedback_message,
                        }
                        st.success("✅ Feedback submitted!")
                else:
                    st.error("Cannot submit feedback without request_id")

    # Summary
    if st.session_state.feedback_data:
        st.markdown("---")
        st.subheader("Feedback Summary")
        positive_count = sum(
            1 for f in st.session_state.feedback_data.values() if f["positive"]
        )
        total_feedback = len(st.session_state.feedback_data)
        st.metric(
            "Feedback Provided",
            f"{total_feedback}/{len(st.session_state.agent_responses)}",
        )
        st.metric(
            "Positive Rate",
            f"{positive_count}/{total_feedback}"
            if total_feedback > 0
            else "N/A",
        )


def render_evaluation():
    """Section 5: LLM Judge Evaluation."""
    st.header("5️⃣ LLM Judge Evaluation")

    if st.session_state.agent_responses is None:
        st.warning("⚠️ Please run the agent first (Section 3)")
        return

    st.markdown("Configure and run automated evaluations")

    # Check for ground truth
    has_ground_truth = False
    if st.session_state.eval_dataset is not None:
        has_ground_truth = (
            "ground_truth" in st.session_state.eval_dataset.columns
        )

    # Metric selection
    st.subheader("Select Metrics")

    col1, col2 = st.columns(2)

    with col1:
        eval_accuracy = st.checkbox(
            "Accuracy (requires ground truth)",
            value=has_ground_truth,
            disabled=not has_ground_truth,
        )

    with col2:
        eval_gpa = st.checkbox(
            "GPA Metrics (logical consistency, efficiency, etc.)",
            value=TRULENS_AVAILABLE,
            disabled=not TRULENS_AVAILABLE,
        )

    if not has_ground_truth:
        st.info(
            "💡 No ground truth available. Accuracy evaluation will be skipped."
        )

    if not TRULENS_AVAILABLE:
        st.warning("⚠️ TruLens not available. GPA evaluations will be skipped.")

    if st.button(
        "🧪 Run Evaluations", use_container_width=True, type="primary"
    ):
        tru_session, tru_provider = get_trulens_components()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in st.session_state.agent_responses.iterrows():
            status_text.text(
                f"Evaluating {idx + 1}/{len(st.session_state.agent_responses)}"
            )

            # Get ground truth if available
            ground_truth = None
            if has_ground_truth and eval_accuracy:
                dataset_row = st.session_state.eval_dataset.iloc[idx]
                ground_truth = dataset_row.get("ground_truth")

            # Run evaluation
            try:
                metrics = evaluate_response(
                    question=row["question"],
                    agent_response_dict=row["response_dict"],
                    ground_truth=ground_truth if eval_accuracy else None,
                    request_id=row["request_id"],
                    agent_name=st.session_state.selected_agent,
                    tru_session=tru_session if eval_gpa else None,
                    tru_provider=tru_provider
                    if (eval_accuracy or eval_gpa)
                    else None,
                )

                result = {
                    "question": row["question"],
                    "agent_response": row["agent_response"],
                    "request_id": row["request_id"],
                }

                # Flatten metrics
                for metric_name, metric_data in metrics.items():
                    result[f"{metric_name}_score"] = metric_data.get(
                        "score", 0.0
                    )
                    result[f"{metric_name}_reason"] = metric_data.get(
                        "reason", ""
                    )

                results.append(result)

            except Exception as e:
                st.error(f"Failed to evaluate question {idx}: {e}")
                results.append({
                    "question": row["question"],
                    "agent_response": row["agent_response"],
                    "request_id": row["request_id"],
                })

            progress_bar.progress(
                (idx + 1) / len(st.session_state.agent_responses)
            )

        st.session_state.evaluation_results = pd.DataFrame(results)
        status_text.text("✅ Evaluation completed!")
        st.success(f"Evaluated {len(results)} responses")

    # Display results
    if st.session_state.evaluation_results is not None:
        st.markdown("---")
        st.subheader("Evaluation Results")

        # Display metrics summary
        score_cols = [
            col
            for col in st.session_state.evaluation_results.columns
            if col.endswith("_score")
        ]

        if score_cols:
            st.markdown("### Metrics Summary")
            metric_cols = st.columns(len(score_cols))

            for idx, col_name in enumerate(score_cols):
                metric_name = (
                    col_name.replace("_score", "").replace("_", " ").title()
                )
                avg_score = st.session_state.evaluation_results[col_name].mean()
                metric_cols[idx].metric(metric_name, f"{avg_score:.3f}")

        # Detailed results
        st.markdown("### Detailed Results")
        st.dataframe(
            st.session_state.evaluation_results, use_container_width=True
        )

        # Download button
        csv = st.session_state.evaluation_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def render_optimization():
    """Section 6: Prompt Optimization."""
    st.header("6️⃣ Prompt Optimization")

    if st.session_state.eval_dataset is None:
        st.warning("⚠️ Please load a dataset first (Section 2)")
        return

    if not st.session_state.selected_agent:
        st.warning("⚠️ Please select an agent first (Section 1)")
        return

    st.markdown("Optimize agent prompts through iterative refinement")

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        num_iterations = st.slider(
            "Number of Iterations", min_value=1, max_value=10, value=3
        )

    with col2:
        temperature = st.slider(
            "Generation Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
        )

    with col3:
        use_manual = st.checkbox("Enable Manual Editing", value=False)

    # Manual prompt editing
    if use_manual:
        st.markdown("---")
        st.subheader("Manual Prompt Editing")

        # Get current instructions
        current_instructions = parse_agent_instructions(
            st.session_state.agent_details
        )

        manual_system = st.text_area(
            "System Instruction",
            value=current_instructions.get("system", ""),
            height=100,
        )

        manual_response = st.text_area(
            "Response Instruction",
            value=current_instructions.get("response", ""),
            height=100,
        )

        manual_orchestration = st.text_area(
            "Orchestration Instruction",
            value=current_instructions.get("orchestration", ""),
            height=100,
        )

        if st.button("✅ Apply Manual Prompts", use_container_width=True):
            agent_manager = get_agent_manager()
            if agent_manager:
                try:
                    instructions = AgentInstructions(
                        system=manual_system,
                        response=manual_response,
                        orchestration=manual_orchestration,
                    )

                    agent_manager.update_agent(
                        name=st.session_state.selected_agent,
                        instructions=instructions,
                    )

                    st.success("✅ Agent updated with manual prompts!")

                    # Refresh agent details
                    details = agent_manager.describe_agent(
                        st.session_state.selected_agent
                    )
                    st.session_state.agent_details = details

                except Exception as e:
                    st.error(f"Failed to update agent: {e}")

    # Automated optimization
    st.markdown("---")
    st.subheader("Automated Optimization")

    # Show button state for debugging
    if st.session_state.optimization_running:
        st.warning(
            "⚠️ Optimization is currently running or was interrupted. Check the Debug Info in the sidebar."
        )
        if st.button("🔓 Reset and Try Again"):
            st.session_state.optimization_running = False
            st.rerun()

    if st.button(
        "🚀 Start Optimization",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.optimization_running,
    ):
        try:
            st.session_state.optimization_running = True
            st.info("Starting optimization... This may take several minutes.")

            agent_manager = get_agent_manager()
            tru_session, tru_provider = get_trulens_components()

            if not agent_manager:
                st.error("Failed to initialize agent manager")
                st.session_state.optimization_running = False
                return

            # Get current instructions
            current_instructions = parse_agent_instructions(
                st.session_state.agent_details
            )

            prev_system = current_instructions.get("system", "")
            prev_response = current_instructions.get("response", "")
            prev_orchestration = current_instructions.get("orchestration", "")
            prev_feedback = None

            # Create status container for live updates
            status_container = st.empty()

            # Optimization loop
            for iteration in range(num_iterations):
                # Update live status
                with status_container.container():
                    st.info(
                        f"🔄 **Running Iteration {iteration + 1}/{num_iterations}**"
                    )
                    st.progress((iteration) / num_iterations)

                st.markdown(f"### Iteration {iteration + 1}/{num_iterations}")

                with st.spinner("Generating prompt variants..."):
                    # Generate new prompts
                    new_system = propose_prompt_variant(
                        role="system instruction",
                        previous_instruction=prev_system,
                        eval_feedback=prev_feedback,
                        temperature=temperature,
                    )

                    new_response = propose_prompt_variant(
                        role="response instruction",
                        previous_instruction=prev_response,
                        eval_feedback=prev_feedback,
                        temperature=temperature,
                    )

                    new_orchestration = propose_prompt_variant(
                        role="orchestration instruction",
                        previous_instruction=prev_orchestration,
                        eval_feedback=prev_feedback,
                        temperature=temperature,
                    )

                # Display generated prompts
                with st.expander("📝 Generated Prompts", expanded=False):
                    st.markdown(f"**System:** {new_system}")
                    st.markdown(f"**Response:** {new_response}")
                    st.markdown(f"**Orchestration:** {new_orchestration}")

                # Update agent
                with st.spinner("Updating agent..."):
                    try:
                        instructions = AgentInstructions(
                            system=new_system,
                            response=new_response,
                            orchestration=new_orchestration,
                        )

                        agent_manager.update_agent(
                            name=st.session_state.selected_agent,
                            instructions=instructions,
                        )
                    except Exception as e:
                        st.error(f"Failed to update agent: {e}")
                        continue

                # Run agent on dataset
                with st.spinner("Running agent on dataset..."):
                    try:
                        thread_id = agent_manager.create_thread(
                            origin_application="optimizer"
                        )

                        responses = []
                        for (
                            idx,
                            row,
                        ) in st.session_state.eval_dataset.iterrows():
                            question = row.get("question", "")
                            if not question:
                                continue

                            response = agent_manager.send_message(
                                agent_name=st.session_state.selected_agent,
                                thread_id=thread_id,
                                message=question,
                            )

                            responses.append({
                                "question": question,
                                "response_dict": response,
                                "request_id": response.get("request_id"),
                            })

                    except Exception as e:
                        st.error(f"Failed to run agent: {e}")
                        continue

                # Evaluate responses
                with st.spinner("Evaluating responses..."):
                    metrics_list = []

                    for idx, response_data in enumerate(responses):
                        ground_truth = None
                        if (
                            "ground_truth"
                            in st.session_state.eval_dataset.columns
                        ):
                            ground_truth = st.session_state.eval_dataset.iloc[
                                idx
                            ].get("ground_truth")

                        try:
                            metrics = evaluate_response(
                                question=response_data["question"],
                                agent_response_dict=response_data[
                                    "response_dict"
                                ],
                                ground_truth=ground_truth,
                                request_id=response_data["request_id"],
                                agent_name=st.session_state.selected_agent,
                                tru_session=tru_session,
                                tru_provider=tru_provider,
                            )
                            metrics_list.append(metrics)
                        except Exception as e:
                            st.warning(f"Failed to evaluate: {e}")
                            metrics_list.append({})

                # Aggregate metrics
                agg_metrics = {}
                agg_reasons = {}

                for metrics in metrics_list:
                    for metric_name, metric_data in metrics.items():
                        if metric_name not in agg_metrics:
                            agg_metrics[metric_name] = []
                            agg_reasons[metric_name] = []

                        agg_metrics[metric_name].append(
                            metric_data.get("score", 0.0)
                        )
                        agg_reasons[metric_name].append(
                            metric_data.get("reason", "")
                        )

                # Average scores
                avg_metrics = {
                    k: sum(v) / len(v) if v else 0.0
                    for k, v in agg_metrics.items()
                }

                # Display iteration results
                if avg_metrics:
                    metric_cols = st.columns(len(avg_metrics))
                    for idx, (metric_name, score) in enumerate(
                        avg_metrics.items()
                    ):
                        metric_cols[idx].metric(
                            metric_name.replace("_", " ").title(),
                            f"{score:.3f}",
                        )

                # Store iteration results
                iteration_result = {
                    "iteration": iteration + 1,
                    "timestamp": datetime.now().isoformat(),
                    "system_instruction": new_system,
                    "response_instruction": new_response,
                    "orchestration_instruction": new_orchestration,
                    "metrics": avg_metrics,
                    "metric_reasons": agg_reasons,
                }

                st.session_state.optimization_history.append(iteration_result)

                # Prepare feedback for next iteration
                prev_system = new_system
                prev_response = new_response
                prev_orchestration = new_orchestration
                prev_feedback = json.dumps(avg_metrics, indent=2)

                st.markdown("---")

            # Final status update
            with status_container.container():
                st.success(f"✅ **Completed all {num_iterations} iterations!**")
                st.progress(1.0)

            st.session_state.optimization_running = False
            st.success("🎉 Optimization completed!")
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            import traceback

            st.code(traceback.format_exc())
        finally:
            st.session_state.optimization_running = False


def render_comparison():
    """Section 7: Results Comparison."""
    st.header("7️⃣ Results Comparison")

    if not st.session_state.optimization_history:
        st.info(
            "💡 No optimization history yet. Run optimization in Section 6 to see results here."
        )
        return

    st.subheader("Optimization History")

    # Create comparison dataframe
    history_data = []
    for result in st.session_state.optimization_history:
        row = {
            "Iteration": result["iteration"],
            "Timestamp": result["timestamp"],
        }

        # Add metrics
        for metric_name, score in result["metrics"].items():
            row[metric_name.replace("_", " ").title()] = score

        history_data.append(row)

    history_df = pd.DataFrame(history_data)

    # Display table
    st.dataframe(history_df, use_container_width=True)

    # Visualization
    if len(history_df) > 1:
        st.markdown("---")
        st.subheader("Metrics Over Iterations")

        # Line chart
        metric_cols = [
            col
            for col in history_df.columns
            if col not in ["Iteration", "Timestamp"]
        ]

        if metric_cols:
            fig = go.Figure()

            for metric_col in metric_cols:
                fig.add_trace(
                    go.Scatter(
                        x=history_df["Iteration"],
                        y=history_df[metric_col],
                        mode="lines+markers",
                        name=metric_col,
                    )
                )

            fig.update_layout(
                title="Metrics Over Iterations",
                xaxis_title="Iteration",
                yaxis_title="Score",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

    # Best iteration
    st.markdown("---")
    st.subheader("Best Iteration")

    metric_cols = [
        col
        for col in history_df.columns
        if col not in ["Iteration", "Timestamp"]
    ]
    if metric_cols:
        # Calculate weighted score (equal weights for all metrics)
        history_df["Combined Score"] = history_df[metric_cols].mean(axis=1)
        best_idx = history_df["Combined Score"].idxmax()
        best_iteration = st.session_state.optimization_history[best_idx]

        st.info(
            f"🏆 Best iteration: **{best_iteration['iteration']}** with combined score **{history_df.iloc[best_idx]['Combined Score']:.3f}**"
        )

        with st.expander("📋 Best Prompts", expanded=True):
            st.markdown(f"**System:** {best_iteration['system_instruction']}")
            st.markdown(
                f"**Response:** {best_iteration['response_instruction']}"
            )
            st.markdown(
                f"**Orchestration:** {best_iteration['orchestration_instruction']}"
            )

        if st.button(
            "✅ Apply Best Prompts to Agent",
            use_container_width=True,
            type="primary",
        ):
            agent_manager = get_agent_manager()
            if agent_manager:
                try:
                    with st.spinner("Updating agent..."):
                        instructions = AgentInstructions(
                            system=best_iteration["system_instruction"],
                            response=best_iteration["response_instruction"],
                            orchestration=best_iteration[
                                "orchestration_instruction"
                            ],
                        )

                        # Show what we're trying to update
                        with st.expander(
                            "📝 Updating with these prompts", expanded=False
                        ):
                            st.markdown(f"**System:** {instructions.system}")
                            st.markdown(
                                f"**Response:** {instructions.response}"
                            )
                            st.markdown(
                                f"**Orchestration:** {instructions.orchestration}"
                            )

                        update_response = agent_manager.update_agent(
                            name=st.session_state.selected_agent,
                            instructions=instructions,
                        )

                        st.success("✅ Agent update API call succeeded!")

                        # Show API response
                        with st.expander(
                            "🔍 Update API Response (Debug)", expanded=False
                        ):
                            st.json(update_response)

                    with st.spinner("Reloading agent details..."):
                        # Refresh agent details
                        details = agent_manager.describe_agent(
                            st.session_state.selected_agent
                        )
                        st.session_state.agent_details = details

                        # Verify the update worked
                        new_instructions = parse_agent_instructions(details)

                        if (
                            new_instructions.get("system")
                            == instructions.system
                        ):
                            st.success(
                                "✅ Verified! Agent details reloaded successfully."
                            )
                            st.info(
                                "💡 Go to Section 1 and click 'Load Agent Details' to see the updated prompts."
                            )
                        else:
                            st.warning(
                                "⚠️ Agent updated but verification shows different values. The API might take a moment to reflect changes."
                            )
                            st.info(
                                "💡 Try clicking 'Load Agent Details' in Section 1 after a few seconds."
                            )

                except Exception as e:
                    st.error(f"❌ Failed to update agent: {e}")
                    import traceback

                    with st.expander("📋 Error Details", expanded=True):
                        st.code(traceback.format_exc())

    # Export
    st.markdown("---")

    export_data = json.dumps(st.session_state.optimization_history, indent=2)
    st.download_button(
        label="📥 Download Full History (JSON)",
        data=export_data,
        file_name=f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# ==================== Main App ====================


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    st.title("🎯 Cortex Agent Prompt Optimizer")
    st.markdown(
        "A comprehensive tool for optimizing Snowflake Cortex Agent prompts through evaluation and iteration"
    )

    st.markdown("---")

    # Render all sections
    render_agent_selection()
    st.markdown("---")

    render_data_loading()
    st.markdown("---")

    render_agent_execution()
    st.markdown("---")

    render_feedback_collection()
    st.markdown("---")

    render_evaluation()
    st.markdown("---")

    render_optimization()
    st.markdown("---")

    render_comparison()


if __name__ == "__main__":
    main()
