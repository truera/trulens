"""
Prompt optimization for Snowflake Cortex Agents using Ray Tune.

This script uses Ray Tune with Optuna search to optimize agent instructions
(system, response, orchestration) across multiple objectives (accuracy,
reasoning quality, conciseness).
"""

import os
from cortex_agent_manager import AgentInstructions
from cortex_agent_manager import CortexAgentManager
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from openai import OpenAI
import pandas as pd
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession
from trulens.core.feedback.selector import Trace
from trulens.feedback import llm_provider
from trulens.providers.cortex.provider import Cortex
from trulens.providers.openai.provider import OpenAI as fOpenAI


# Load environment variables
load_dotenv()


# ==================== Configuration ====================

# Agent configuration
AGENT_NAME = "PROMPT_OPTIM_AGENT"

# Evaluation dataset
EVAL_DATA = [
    # Basic facts
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote 1984?", "answer": "George Orwell"},
    {"question": "What is 12 * 8?", "answer": "96"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    # Multi-step reasoning
    {
        "question": "If a train travels 60 miles in 45 minutes, how many miles does it travel in 2 hours?",
        "answer": "160 miles",
    },
    {
        "question": "Alice has twice as many apples as Bob. Bob has 3 fewer apples than Carol who has 12 apples. How many apples does Alice have?",
        "answer": "18 apples",
    },
    # Comparative analysis
    {
        "question": "What is the main difference between a democracy and a republic?",
        "answer": "A democracy is direct rule by the people, while a republic is representative government through elected officials",
    },
    {
        "question": "How does mitosis differ from meiosis?",
        "answer": "Mitosis produces two identical diploid cells, while meiosis produces four genetically diverse haploid cells",
    },
    # Contextual understanding
    {
        "question": "Why did the chicken cross the road?",
        "answer": "To get to the other side",
    },
    {
        "question": "What does 'break the ice' mean?",
        "answer": "To start a conversation or ease tension in a social situation",
    },
    # Technical explanations
    {
        "question": "What is the time complexity of binary search?",
        "answer": "O(log n)",
    },
    {
        "question": "Explain what REST API stands for.",
        "answer": "Representational State Transfer Application Programming Interface",
    },
    # Scientific concepts
    {
        "question": "What is photosynthesis?",
        "answer": "The process by which plants convert light energy into chemical energy to produce glucose from carbon dioxide and water",
    },
    {
        "question": "What causes seasons on Earth?",
        "answer": "The tilt of Earth's axis as it orbits the sun",
    },
    # Historical context
    {"question": "In what year did World War II end?", "answer": "1945"},
    {
        "question": "Who was the first person to walk on the moon?",
        "answer": "Neil Armstrong",
    },
    # Problem-solving
    {
        "question": "If you have a 3-gallon jug and a 5-gallon jug, how can you measure exactly 4 gallons?",
        "answer": "Fill the 5-gallon jug, pour into the 3-gallon jug, leaving 2 gallons. Empty the 3-gallon jug, pour the 2 gallons into it, then fill the 5-gallon jug again and pour 1 gallon into the 3-gallon jug to top it off, leaving 4 gallons in the 5-gallon jug",
    },
    {
        "question": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "answer": "42",
    },
]

# Optimization configuration
NUM_SAMPLES = 3  # Number of optimization trials
TOP_K = 3  # Number of Pareto-optimal configs to display

# MLflow configuration
MLFLOW_URI = "file:///tmp/mlruns"
EXPERIMENT_NAME = "agent_prompt_optimization"
PARENT_RUN_NAME = "agent_parent_run"

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

TRIED_INSTRUCTIONS = {
    "system_instructions": [],
    "response_instructions": [],
    "orchestration_instructions": [],
    "evaluation_feedback": [],
}

# TODO: consider writing a GEPA adapter to factor in the full trace and feedback for more advanced reflective prompt optimization (see dspy/../gepa.py)


# ==================== Helper Functions ====================


def get_trulens_eval_components():
    """
    Initialize TruLens components (TruSession and OpenAI provider).

    This function is called inside the objective function to avoid Ray serialization issues.
    Each Ray worker will create its own instances.

    Returns:
        Tuple of (TruSession, OpenAI provider)
    """
    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PAT"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

    snowpark_session = Session.builder.configs(connection_params).create()
    tru_session = TruSession(
        SnowflakeConnector(snowpark_session=snowpark_session)
    )
    # cortex_provider = fCortex(
    #     snowpark_session=snowpark_session, model_engine="auto"
    # )
    openai_provider = fOpenAI(model_engine=LLM_MODEL)

    return tru_session, openai_provider


def get_agent_manager() -> CortexAgentManager:
    """
    Initialize Cortex Agent Manager from environment variables.

    Returns:
        CortexAgentManager instance
    """
    return CortexAgentManager(
        account_url=os.getenv("SNOWFLAKE_ACCOUNT_URL"),
        auth_token=os.getenv("SNOWFLAKE_PAT"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


def setup_agent(agent_manager: CortexAgentManager, agent_name: str):
    """
    Create agent if it doesn't exist, or verify it exists.

    Args:
        agent_manager: CortexAgentManager instance
        agent_name: Name of the agent to create/verify
    """
    try:
        # Try to describe the agent to see if it exists
        agent_manager.describe_agent(agent_name)
        print(f"Agent '{agent_name}' already exists. Using existing agent.")
    except Exception:
        # Agent doesn't exist, create it
        print(f"Creating agent '{agent_name}'...")

        initial_instructions = AgentInstructions(
            system="",
            response="",
            orchestration="",
        )

        agent_manager.create_agent(
            name=agent_name,
            instructions=initial_instructions,
            tools=[],
            tool_resources={},
            create_mode="ifNotExists",
            comment="Agent for prompt optimization experiments",
        )
        print(f"Agent '{agent_name}' created successfully.")


def extract_agent_response(response: dict) -> str:
    """
    Extract text response from agent message.

    Args:
        response: Agent response dictionary

    Returns:
        Extracted text content
    """
    message_obj = response.get("message", {})
    content = message_obj.get("content", [])

    for content_item in content:
        if content_item.get("type") == "text":
            return content_item.get("text", "")

    return ""


# ==================== LLM-Based Generation & Evaluation ====================


# TODO: consider using Cortex AI_COMPLETE endpoint?
# TODO: fine-tune prompts for each role (system, response, orchestration)
# TODO: add logic to propose variants for tool descriptions as well
def propose_variant(
    role: str,
    previous_instruction: str = None,
    eval_feedback: dict = None,
) -> str:
    """
    Use LLM to generate a variant of an instruction.

    Args:
        role: Type of instruction (e.g., "system instruction")
        previous: Previous instruction to use as reference

    Returns:
        Generated instruction text
    """
    client = OpenAI()

    base_prompt = f"Generate a new {role} for an existing AI agent. Keep it concise and to the point."
    if previous_instruction:
        base_prompt += f"""
            Use the previous {role} as a reference: ```{previous_instruction}```.

            """
    if eval_feedback:
        base_prompt += f"""
        The previous {role}'s evaluation feedback was: ```{eval_feedback}```.
        """

    base_prompt += f"""
    The new {role} should be different from the previous one, and take into account the evaluation feedback to score better.
    Focus on clarity, factual accuracy, usefulness, and conciseness.
    """

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer."},
            {"role": "user", "content": base_prompt},
        ],
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


def evaluate_agent_response(
    question: str,
    agent_response: dict,
    ground_truth: str,
    tru_session: TruSession,
    tru_provider: llm_provider.LLMProvider,
) -> dict:
    """
    Evaluate agent response using LLM-as-a-Judge metrics. This includes
    ground-truth based evals and reference-free GPA (Goal-Plan-Action) evals via TruLens.

    Args:
        question: Original question
        agent_response: Cortex Agent REST API Run endpoint response
        ground_truth: Expected answer
        tru_session: TruLens session for retrieving observability events
        tru_provider: Cortex provider for running evaluations

    Returns:
        Dictionary with nested GPA metrics:
            - accuracy: {score: float, reason: str}
            - logical_consistency: {score: float, reason: str}
            - execution_efficiency: {score: float, reason: str}
            - plan_adherence: {score: float, reason: str}
            - plan_quality: {score: float, reason: str}
    """

    metrics = {}

    # ==================== Ground-truth based evaluations ====================
    # Extract agent response
    agent_response_text = extract_agent_response(agent_response)

    # Accuracy eval

    accuracy_prompt = f"""
    You are an impartial evaluator.

    Question: {question}
    Agent Response: {agent_response_text}
    Ground Truth: {ground_truth}

    Score the agent response on its accuracy given the question and the ground truth.

    Please evaluate using the following template:
    Criteria: <Provide the criteria for this evaluation, restating the
    criteria you are using to evaluate>
    Supporting Evidence: <Provide your reasons for scoring based on the
    listed criteria step by step. Tie it back to the evaluation being
    completed.>
    Score: <The score based on the given criteria>
    Please respond using the entire template above.
    """

    accuracy_score, accuracy_reason = tru_provider.generate_score_and_reasons(
        system_prompt=accuracy_prompt,
        temperature=0.0,
    )

    metrics["accuracy"] = {
        "score": accuracy_score,
        "reason": accuracy_reason["reason"],
    }

    # client = OpenAI()

    # judge_prompt = f"""
    # You are an impartial evaluator.

    # Question: {question}
    # Agent Response: {agent_response}
    # Ground Truth: {ground_truth}

    # Score the answer on three axes:
    # 1. Accuracy (0-1)
    # 2. Reasoning (0-1)
    # 3. Conciseness (0-1)

    # Respond in JSON format like:
    # {
    # "accuracy": {"score": 0.9, "reason": "The agent's response is accurate."},
    # "reasoning": {"score": 0.8, "reason": "The agent's response is well-reasoned."},
    # "conciseness": {"score": 0.7, "reason": "The agent's response is concise."}
    # }
    # """

    # resp = client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=[
    #         {"role": "system", "content": "You are a strict evaluator."},
    #         {"role": "user", "content": judge_prompt},
    #     ],
    #     temperature=0.0,
    # )
    # try:
    #     metrics = json.loads(resp.choices[0].message.content.strip())
    # except Exception:
    #     metrics = {"accuracy": 0.0, "reasoning": 0.0, "conciseness": 0.0}
    # return metrics

    # ==================== Reference-free GPA evaluations ====================
    # Extract request ID for retrieving observability events
    request_id = agent_response.get("request_id", None)

    # Get events from Snowflake AI Observability Event Table
    events_df = tru_session.get_events(
        app_name=AGENT_NAME, app_version=None, record_ids=[request_id]
    )
    trace = Trace()
    trace.events = events_df

    # Define GPA evals functions - each returns tuple[float, dict] where dict contains {"reason": str}
    gpa_evaluators = {
        "logical_consistency": tru_provider.logical_consistency_with_cot_reasons,
        "execution_efficiency": tru_provider.execution_efficiency_with_cot_reasons,
        "plan_adherence": tru_provider.plan_adherence_with_cot_reasons,
        "plan_quality": tru_provider.plan_quality_with_cot_reasons,
    }

    # Run all GPA evals and organize results
    for metric_name, eval_func in gpa_evaluators.items():
        score, reason_dict = eval_func(trace=trace)
        metrics[metric_name] = {
            "score": score,
            "reason": reason_dict["reason"],
        }

    return metrics


# ==================== Ray Tune Objective ====================


def objective(config):
    """
    Objective function for Ray Tune optimization.

    Generates new instruction variants, updates the agent, evaluates on EVAL_DATA,
    and reports metrics back to Ray Tune.

    Args:
        config: Configuration dictionary from Ray Tune search space
    """
    metric_names = [
        # Ground-truth based evals
        "accuracy",
        # "reasoning",
        # "conciseness",
        # GPA evals
        "logical_consistency",
        "execution_efficiency",
        "plan_adherence",
        "plan_quality",
    ]

    # prev_sys = config.get("system_instructions")
    # prev_resp = config.get("response_instructions")
    # prev_orch = config.get("orchestration_instructions")

    last_tried_sys_instr = (
        TRIED_INSTRUCTIONS["system_instructions"][-1]
        if len(TRIED_INSTRUCTIONS["system_instructions"]) > 0
        else None
    )
    last_tried_resp_instr = (
        TRIED_INSTRUCTIONS["response_instructions"][-1]
        if len(TRIED_INSTRUCTIONS["response_instructions"]) > 0
        else None
    )
    last_tried_orch_instr = (
        TRIED_INSTRUCTIONS["orchestration_instructions"][-1]
        if len(TRIED_INSTRUCTIONS["orchestration_instructions"]) > 0
        else None
    )
    last_tried_eval_feedback = (
        TRIED_INSTRUCTIONS["evaluation_feedback"][-1]
        if len(TRIED_INSTRUCTIONS["evaluation_feedback"]) > 0
        else None
    )

    # Generate new instruction variants using LLM
    curr_sys_instr = propose_variant(
        role="system instruction",
        previous_instruction=last_tried_sys_instr,
        eval_feedback=last_tried_eval_feedback,
    )
    curr_resp_instr = propose_variant(
        role="response instruction",
        previous_instruction=last_tried_resp_instr,
        eval_feedback=last_tried_eval_feedback,
    )
    curr_orch_instr = propose_variant(
        role="orchestration instruction",
        previous_instruction=last_tried_orch_instr,
        eval_feedback=last_tried_eval_feedback,
    )

    # Initialize TruLens components (avoid global variables for Ray serialization)
    tru_session, tru_provider = get_trulens_eval_components()

    # Update agent with new instructions
    instructions = AgentInstructions(
        system=curr_sys_instr,
        response=curr_resp_instr,
        orchestration=curr_orch_instr,
    )

    try:
        AGENT_MANAGER.update_agent(
            name=AGENT_NAME,
            instructions=instructions,
        )
    except Exception as e:
        print(f"Failed to update agent: {e}")
        # Report zeros if agent update fails
        tune.report({
            **{name: 0.0 for name in metric_names},
            "system_instructions": curr_sys_instr,
            "response_instructions": curr_resp_instr,
            "orchestration_instructions": curr_orch_instr,
        })
        return

    # Create a thread for this trial
    try:
        thread_id = AGENT_MANAGER.create_thread(
            origin_application="prompt_optim"
        )
    except Exception as e:
        print(f"Failed to create thread: {e}")
        tune.report({
            **{name: 0.0 for name in metric_names},
            "system_instructions": curr_sys_instr,
            "response_instructions": curr_resp_instr,
            "orchestration_instructions": curr_orch_instr,
        })
        return

    agg_metric_scores = {name: 0.0 for name in metric_names}
    agg_metric_reasons = {name: "" for name in metric_names}

    for item in EVAL_DATA:
        try:
            # Send message to agent
            agent_response = AGENT_MANAGER.send_message(
                agent_name=AGENT_NAME,
                thread_id=thread_id,
                message=item["question"],
            )

            # Evaluate the answer
            metrics = evaluate_agent_response(
                question=item["question"],
                agent_response=agent_response,
                ground_truth=item["answer"],
                tru_session=tru_session,
                tru_provider=tru_provider,
            )

            # Aggregate scores from nested structure
            for metric_name in metric_names:
                # Aggregate scores
                agg_metric_scores[metric_name] += metrics.get(
                    metric_name, {}
                ).get("score", 0.0)
                # Append reason to aggregate reasons, separated by a newline
                agg_metric_reasons[metric_name] += (
                    f"\n{metrics.get(metric_name, {}).get('reason', '')}"
                )

        except Exception as e:
            print(f"Error evaluating question '{item['question']}': {e}")
            # Add zeros for failed evaluations
            for metric_name in metric_names:
                agg_metric_scores[metric_name] += 0.0
                agg_metric_reasons[metric_name] += ""

    # Average scores over dataset
    for metric_name in metric_names:
        agg_metric_scores[metric_name] /= len(EVAL_DATA)

    # Report to Ray Tune (auto-logged by MLflow)
    tune.report({
        **agg_metric_scores,
        "system_instructions": curr_sys_instr,
        "response_instructions": curr_resp_instr,
        "orchestration_instructions": curr_orch_instr,
    })

    # Log to tried instructions
    TRIED_INSTRUCTIONS["system_instructions"].append(curr_sys_instr)
    TRIED_INSTRUCTIONS["response_instructions"].append(curr_resp_instr)
    TRIED_INSTRUCTIONS["orchestration_instructions"].append(curr_orch_instr)
    TRIED_INSTRUCTIONS["evaluation_feedback"].append(agg_metric_reasons)


# ==================== Pareto Analysis ====================


def is_pareto_efficient(costs=np.ndarray) -> np.ndarray:
    """
    Identify Pareto-efficient (non-dominated) points.

    Args:
        costs: Numpy array of shape (n_points, n_objectives)

    Returns:
        Boolean array indicating which points are Pareto-efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient


# ==================== Visualization ====================


def plot_pareto_frontier(df: pd.DataFrame, pareto_mask: np.ndarray):
    """
    Plot 3D Pareto frontier showing optimal trade-offs (accuracy, logical consistency, execution efficiency).

    Args:
        df: DataFrame with trial results
        pareto_mask: Boolean array indicating Pareto-optimal points
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Use first 3 metrics for 3D visualization
    metric_cols = [
        "accuracy",
        "logical_consistency",
        "execution_efficiency",
    ]
    all_points = df[metric_cols].values
    pareto_points = all_points[pareto_mask]

    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        pareto_points[:, 2],
        c="blue",
        alpha=0.3,
        label="All trials",
    )
    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        pareto_points[:, 2],
        c="red",
        label="Pareto front",
        s=80,
    )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Logical Consistency")
    ax.set_zlabel("Execution Efficiency")
    ax.set_title(
        "Pareto Frontier (Accuracy, Logical Consistency, Execution Efficiency)"
    )
    ax.legend()
    plt.show()


# ==================== Main Optimization Loop ====================


def run_optimizer():
    """
    Run multi-objective optimization using Ray Tune + Optuna.

    Optimizes agent instructions to maximize accuracy, reasoning quality,
    and conciseness. Displays Pareto-optimal results and plots the frontier.
    """

    search_space = {
        "system_instructions": tune.choice([None]),
        "response_instructions": tune.choice([None]),
        "orchestration_instructions": tune.choice([None]),
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Parent run to group trials
    with mlflow.start_run():
        mlflow_logger = MLflowLoggerCallback(
            tracking_uri=MLFLOW_URI,
            experiment_name=EXPERIMENT_NAME,
            tags={"parent_run_name": PARENT_RUN_NAME},
            save_artifact=True,
        )

        # Multi-objective optimization with GPA metrics
        algo = OptunaSearch(
            metric=[
                "accuracy",
                "logical_consistency",
                "execution_efficiency",
                "plan_adherence",
                "plan_quality",
            ],
            mode=[
                "max",
                "max",
                "max",
                "max",
                "max",
            ],
        )

        tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=algo,
                num_samples=NUM_SAMPLES,
            ),
            run_config=tune.RunConfig(
                name="mlflow",
                callbacks=[mlflow_logger],
            ),
        )

        results = tuner.fit()

    # Compute Pareto frontier
    df = results.get_dataframe()
    metric_cols = [
        "accuracy",
        "logical_consistency",
        "execution_efficiency",
        "plan_adherence",
        "plan_quality",
    ]
    metrics_array = df[metric_cols].values
    pareto_mask = is_pareto_efficient(metrics_array)
    pareto_df = df[pareto_mask]

    print(f"Top-{TOP_K} Pareto-optimal configs:")
    top_k = pareto_df.sort_values(by=metric_cols, ascending=False).head(TOP_K)
    for idx, row in top_k.iterrows():
        print(
            f"Accuracy={row['accuracy']:.2f}, "
            f"Logical Consistency={row['logical_consistency']:.2f}, "
            f"Execution Efficiency={row['execution_efficiency']:.2f}, "
            f"Plan Adherence={row['plan_adherence']:.2f}, "
            f"Plan Quality={row['plan_quality']:.2f}"
        )
        print("-" * 50)
        print(f"System: {row['system_instructions']}")
        print("-" * 50)
        print(f"Response: {row['response_instructions']}")
        print("-" * 50)
        print(f"Orchestration: {row['orchestration_instructions']}")
        print("-" * 50)
    # plot_pareto_frontier(df, pareto_mask)


# ==================== Main Entry Point ====================


if __name__ == "__main__":
    # Initialize global agent manager (used in objective function)
    AGENT_MANAGER = get_agent_manager()
    # Ensure agent exists before optimization
    print("=" * 60)
    print("Setting up agent...")
    setup_agent(AGENT_MANAGER, AGENT_NAME)
    print("=" * 60)

    # Run optimizer
    print("Running optimizer...")
    run_optimizer()
    print("=" * 60)

    # ======================== SCRATCH PAD ======================

    # thread_id = AGENT_MANAGER.create_thread(origin_application="prompt_optim")

    # agent_response = AGENT_MANAGER.send_message(
    #     agent_name=AGENT_NAME,
    #     thread_id=thread_id,
    #     message="What is the meaning of life?",
    # )

    # print(agent_response)

    # request_id = agent_response.get("request_id", None)

    # tru_session, tru_provider = get_trulens_eval_components()

    # # Get events from Snowflake AI Observability Event Table
    # events_df = tru_session.get_events(
    #     app_name=AGENT_NAME,
    #     app_version=None,
    #     record_ids=["fc981cd1-e07c-4506-a728-a6639c3d7c65"],
    # )
    # trace = Trace()
    # trace.events = events_df

    # print(trace)

    # score, reason_dict = tru_provider.logical_consistency_with_cot_reasons(
    #     trace=trace
    # )

    # print(f"Logical Consistency Score: {score}")
    # print(f"Logical Consistency Reason: {reason_dict['reason']}")
