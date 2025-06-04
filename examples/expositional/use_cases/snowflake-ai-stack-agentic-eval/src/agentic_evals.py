import os
from typing import Literal, Dict, Tuple
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.core.otel.instrument import instrument
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
from trulens.feedback import prompts


from langgraph.types import Command
from langchain_core.messages import HumanMessage

from src.util import append_to_step_trace

import json

import streamlit as st


def extract_reason(val):
    if isinstance(val, dict) and 'reason' in val:
        return val['reason']
    if isinstance(val, dict) and 'reasons' in val:
        return val['reasons']
    return str(val)

@instrument(
    name="research_eval_node",
    description="Evaluates the research agent's output based on context relevance, groundedness, and answer relevance.",
    attributes=lambda ret, exception, *args, **kwargs: {
        f"{BASE_SCOPE}.execution_trace": json.dumps(args[0].get("execution_trace", {})),
        f"{BASE_SCOPE}.user_query": args[0].get("user_query", {}),
        f"{BASE_SCOPE}.current_step": args[0].get("current_step", 0),
        f"{BASE_SCOPE}.last_reason": args[0].get("last_reason", ""),
            f"{BASE_SCOPE}.research_eval_node_response": ret.update["messages"][
            -1
        ].content
        if hasattr(ret, "update")
        else json.dumps(ret, indent=4, sort_keys=True),
    },
)
def research_eval_node(state) -> Command[Literal["orchestrator"]]:
    query = state.get("user_query")
    context_list = state.get("execution_trace")[state.get("current_step")][-1]["tool_calls"]
    response = state.get("execution_trace")[state.get("current_step")][-1]["output"]

    provider = OpenAI(model_engine=os.environ.get("LLM_MODEL_NAME"))

    with st.spinner("Evaluating research..."):
        context_rel_score, context_rel_reason = provider.context_relevance_with_cot_reasons(
                question=query,
                context=context_list,
            )
        grounded_score, grounded_reason = provider.groundedness_measure_with_cot_reasons(
            source=" ".join(context_list),
            statement=response
        )
        answer_score, answer_reason = provider.relevance_with_cot_reasons(prompt=query, response=response)

    goto = "orchestrator"
    parsed_eval = {
        "context_relevance": {
            "score": context_rel_score,
            "reason": context_rel_reason
        },
        "groundedness": {
            "score": grounded_score,
            "reason": grounded_reason
        },
        "answer_relevance": {
            "score": answer_score,
            "reason": answer_reason
        }
    }

    # Create new entry for execution_trace
    eval_entry = {
        "step": state.get("current_step"),
        "agent": "research_eval",
        "metrics": parsed_eval,
    }
    # Build a natural language summary
    summary = (
        f"Research Evaluation:\n"
        f"- Context Relevance: {float(str(parsed_eval['context_relevance']['score']))*3}/3 — {extract_reason(parsed_eval['context_relevance']['reason'])}\n"
        f"- Groundedness: {float(str(parsed_eval['groundedness']['score']))*3}/3 — {extract_reason(parsed_eval['groundedness']['reason'])}\n"
        f"- Answer Relevance: {float(str(parsed_eval['answer_relevance']['score']))*3}/3 — {extract_reason(parsed_eval['answer_relevance']['reason'])}"
    )
    eval_msg = HumanMessage(content=summary, name="research_eval")

    return Command(
        update={
            "messages": [eval_msg],
            "execution_trace": append_to_step_trace(state, state.get("current_step"), eval_entry)
        },
        goto=goto
    )


class CustomChartEval(OpenAI):
    def chart_accuracy_with_cot_reasons(self, code: str, context: str) -> Tuple[float, Dict]:
        system_prompt = f"""
        Use the following rubric to evaluate chart accuracy based on context:
        0: The chart does not reflect the data, plots incorrect values/relationships, or plots hypothetical data.
        1: The chart has significant errors (e.g., wrong labels, major mismatches in data) but shows some partial attempt.
        2: The chart is mostly correct with minor, non-critical errors.
        3: The chart is completely accurate: correct data, correct relationships, correct calculations.
        """
        user_prompt = f""" Please score the code based on the context. Code: {code}. \n Context: {context}. \n {prompts.COT_REASONS_TEMPLATE}
        """
        return self.generate_score_and_reasons(system_prompt=system_prompt, user_prompt=user_prompt, min_score_val = 0, max_score_val = 3)
    def chart_formatting_with_cot_reasons(self, code: str) -> Tuple[float, Dict]:
        system_prompt = f"""
        Use the following rubric to evaluate chart formatting:
        0: The chart is poorly formatted: missing important elements like titles, axis labels, or has unreadable text.
        1: Basic elements are present but formatting is cluttered, confusing, or difficult to read.
        2: The chart is mostly clean and readable with only minor formatting issues.
        3: The chart is well-formatted with clear titles, labeled axes with reasonable units, readable scales, appropriate legends, and an overall clean presentation.
        """
        user_prompt = f""" Please score the code. Code: {code}. \n {prompts.COT_REASONS_TEMPLATE}
        """
        return self.generate_score_and_reasons(system_prompt=system_prompt, user_prompt=user_prompt, min_score_val = 0, max_score_val = 3)
    def chart_relevance_with_cot_reasons(self, code: str, query: str, response: str) -> Tuple[float, Dict]:
        system_prompt = f"""
        Use the following rubric to evaluate chart and response relevance to query:
        0: The chart and response are irrelevant to the user's request or misinterprets the task.
        1: Either the chart or the response partially addresses the user query but misses major elements or misunderstands key parts.
        2: The chart and response mostly answers the user query but could be more precise or complete.
        3: The chart and response directly and fully answers the user query with an appropriate and complete visual representation.
        """
        user_prompt = f""" Please score the relevance of the chart and response to the query. Code: {code}. \n Response: {response}. Query: {query} \n {prompts.COT_REASONS_TEMPLATE}
        """
        return self.generate_score_and_reasons(system_prompt=system_prompt, user_prompt=user_prompt, min_score_val = 0, max_score_val = 3)

chart_provider = CustomChartEval(model_engine=os.environ.get("LLM_MODEL_NAME"))#, base_url="https://api.openai.com/v1")

@instrument(
    name="chart_eval_node",
    description="Evaluates the chart generated by the chart generation node.",
    attributes=lambda ret, exception, *args, **kwargs: {
        f"{BASE_SCOPE}.execution_trace": json.dumps(args[0].get("execution_trace", {})),
        f"{BASE_SCOPE}.user_query": args[0].get("user_query", {}),
        f"{BASE_SCOPE}.current_step": args[0].get("current_step", 0),
        f"{BASE_SCOPE}.last_reason": args[0].get("last_reason", ""),
            f"{BASE_SCOPE}.chart_eval_node_response": ret.update["messages"][
            -1
        ].content
        if hasattr(ret, "update")
        else json.dumps(ret, indent=4, sort_keys=True),
    },
)
def chart_eval_node(state) -> Command[Literal["orchestrator"]]:
    context = state.get("execution_trace")[state.get("current_step")-1][-2]["output"]
    code = state.get("execution_trace")[state.get("current_step")][-1]["code"]
    query = state.get("user_query")
    response = state.get("execution_trace")[state.get("current_step")][-1]["output"]

    with st.spinner("Evaluating chart..."):
        accuracy_rel_score, accuracy_rel_reason = chart_provider.chart_accuracy_with_cot_reasons(code=code, context=context)
        formatting_score, formatting_reason = chart_provider.chart_formatting_with_cot_reasons(code=code)
        relevance_score, relevance_reason = chart_provider.chart_relevance_with_cot_reasons(code=code, query=query, response=response)

    parsed_eval = {
        "accuracy": {
            "score": accuracy_rel_score,
            "reason": accuracy_rel_reason
        },
        "formatting": {
            "score": formatting_score,
            "reason": formatting_reason
        },
        "answer_relevance": {
            "score": relevance_score,
            "reason": relevance_reason
        }
    }

    eval_entry = {
        "step": state.get("current_step"),
        "agent": "chart_eval",
        "metrics": parsed_eval,
    }
    # Build a natural language summary
    summary = (
        f"Chart Evaluation:\n"
        f"- Accuracy: {float(str(parsed_eval['accuracy']['score']))*3}/3 — {extract_reason(parsed_eval['accuracy']['reason'])}\n"
        f"- Formatting: {float(str(parsed_eval['formatting']['score']))*3}/3 — {extract_reason(parsed_eval['formatting']['reason'])}\n"
        f"- Answer Relevance: {float(str(parsed_eval['answer_relevance']['score']))*3}/3 — {extract_reason(parsed_eval['answer_relevance']['reason'])}"
    )
    eval_msg = HumanMessage(content=summary, name="chart_eval")
    goto = "orchestrator"
    return Command(
        update={
            "messages": [eval_msg],
            "execution_trace": append_to_step_trace(state, state.get("current_step"), eval_entry)
        },
        goto=goto
    )


class CustomTrajEval(OpenAI):
    def traj_execution_with_cot_reasons(self, trace) -> Tuple[float, Dict]:
        system_prompt = f"""
        Use the following rubric to evaluate the execution trace of the system:
        0: Agents made wrong or unnecessary calls. Critical steps were skipped or repeated without purpose. Generated outputs were off-topic, hallucinated, or contradictory. Confusion between agent roles. User goal was not meaningfully addressed.
        1: Several unnecessary or misordered agent/tool use. Some factual errors or under-specified steps. Redundant or partially irrelevant tool calls. Weak or ambiguous agent outputs at one or more steps
        2: Some minor inefficiencies or unclear transitions. Moments of stalled progress, but ultimately resolved. The agents mostly fulfilled their roles, and the conversation mostly fulfilled answering the query.
        3: Agent handoffs were well-timed and logical. Tool calls were necessary, sufficient, and accurate. No redundancies, missteps, or dead ends. Progress toward the user query was smooth and continuous. No hallucination or incorrect outputs
        """
        user_prompt = f""" Please score the execution trace. Execution Trace (possible in json str format): {trace}. \n
        {prompts.COT_REASONS_TEMPLATE}
        """
        return self.generate_score_and_reasons(system_prompt=system_prompt, user_prompt=user_prompt, min_score_val = 0, max_score_val = 3)

def create_traj_eval(provider) -> Feedback:
    return (
        Feedback(provider.traj_execution_with_cot_reasons, name="Trajectory Evaluation")
        .on({
            "trace": Selector(
                span_type="ORCHESTRATOR_NODE",
                span_attribute=f"{BASE_SCOPE}.execution_trace",
                collect_list=True,
            ),
        })
    )
