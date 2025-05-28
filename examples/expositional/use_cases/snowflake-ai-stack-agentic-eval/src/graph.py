from typing import Iterator, Optional, List, Dict, Any, Annotated, Literal

from IPython.display import Image
from IPython.display import display
from langchain.load.dump import dumps
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import SpanAttributes
import json
from src.agentic_evals import chart_eval_node, traj_eval_node
from src.util import State, append_to_step_trace
import streamlit as st


def build_graph(search_max_results: int, llm_model: str, reasoning_model: str) -> StateGraph:
    llm = ChatOpenAI(
        model=llm_model,
    )
    reasoning_llm = ChatOpenAI(model=reasoning_model)

    def make_system_prompt(suffix: str) -> str:
        return (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK, another assistant with different tools "
            " will help where you left off. Execute what you can to make progress."
            " If you or any of the other assistants have the final answer or deliverable,"
            " prefix your response with FINAL ANSWER so the team knows to stop."
            f"\n{suffix}"
        )

    tavily_tool = TavilySearchResults(max_results=search_max_results)

    # Warning: This executes code locally, which can be unsafe when not sandboxed
    repl = PythonREPL()

    @tool
    @instrument(
        span_type="PYTHON_REPL_TOOL",
        attributes={
            f"{BASE_SCOPE}.python_tool_input_code": "code",
        },
    )
    def python_repl_tool(
        code: Annotated[
            str, "The python code to execute to generate your chart."
        ],
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""

        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        result_str = (
            f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        )
        return (
            result_str
            + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )


    # Research agent and node
    research_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        prompt=make_system_prompt(f"""
            You are the Researcher. You can ONLY perform research by using the provided search tool (tavily_tool).
            Your only job is to find and return exact numerical values (no approximations) based on the user's request, formatted in a clean structured format.
            Only include values explicitly stated in your sources.
            Do NOT infer missing values.
            When you have found the necessary information, end your output.
            Do NOT attempt to take further actions.
        """),
    )

    @instrument(
        span_type="RESEARCH_NODE",
        attributes=lambda ret, exception, *args, **kwargs: {
            f"{BASE_SCOPE}.execution_trace": args[0].get("execution_trace", {}),
            f"{BASE_SCOPE}.user_query": args[0].get("user_query", {}),
            f"{BASE_SCOPE}.current_step": args[0].get("current_step", 0),
            f"{BASE_SCOPE}.last_reason": args[0].get("last_reason", ""),
             f"{BASE_SCOPE}.research_node_response": ret.update["messages"][
                -1
            ].content
            if hasattr(ret, "update")
            else json.dumps(ret, indent=4, sort_keys=True),
        },
    )
    def research_node(
        state: State,
    ) -> Command[Literal["orchestrator"]]:
        with st.spinner("Researcher step..."):
            result = research_agent.invoke(state)
        goto = "orchestrator"
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )

        # Create new entry for execution_trace
        new_context = {
            "step": state.get("current_step"),
            "agent": "researcher",
            "output": result["messages"][-1].content,
            "tool_calls": [m.content for m in result["messages"] if isinstance(m, ToolMessage)],
        }

        return Command(
            update={
                "messages": result["messages"],
                "execution_trace": append_to_step_trace(state, state.get('current_step'), new_context),
            },
            goto=goto
        )


    def orchestrator_prompt(state):
        step = state.get("current_step", 0)
        if step == 0:
            step_evals = []
        else:
            step_evals = state.get("execution_trace")[step][-1]["metrics"]
        return HumanMessage(content=f"""
            You are the Orchestrator in a multi-agent system with three agents: `researcher`, `chart_generator`, and `traj_eval`.

            Your responsibilities:
            1. Decide which agent to call next (`goto`: one of `"researcher"`, `"chart_generator"`, `"traj_eval"`).
            2. Provide a 1-sentence justification for your decision (`reason`: string).

            Use the evaluation scores, execution trace, and current step to inform your decision.

            Guidelines:
            - If the current step is partially completed but more progress can be made, call the agent again.
            - If performance has improved and the current step has been completed (even across multiple agent calls), move to the next agent.
            - If multiple failures have occurred with no improvement, move on to the next agent with your existing knowledge.

            Inputs:
            - User query: {state.get("user_query", "[Missing]")}
            - Evaluations for previous step: {step_evals}
            - Previous step: {step}

            Respond in the following JSON format **with no additional explanation**:

            {{
            "goto": "<agent_name>",
            "reason": "<1 sentence explanation>"
            }}
        """)


    @instrument(
        span_type="ORCHESTRATOR_NODE",
        attributes=lambda ret, exception, *args, **kwargs: {
            f"{BASE_SCOPE}.execution_trace": args[0].get("execution_trace", {}),
            f"{BASE_SCOPE}.user_query": args[0].get("user_query", {}),
            f"{BASE_SCOPE}.current_step": args[0].get("current_step", 0),
            f"{BASE_SCOPE}.last_reason": args[0].get("last_reason", ""),
        },
    )
    def orchestrator_node(
        state: State,
    ) -> Command[Literal["researcher", "chart_generator", "traj_eval"]]:
        full_prompt = [orchestrator_prompt(state)]
        with st.spinner("Orchestrator step..."):
            result = reasoning_llm.invoke(full_prompt)

        try:
            parsed = json.loads(result.content)
            goto = parsed["goto"]
            reason = parsed["reason"]
        except Exception as e:
            raise ValueError(f"Invalid orchestrator JSON: {result.content}") from e

        step = state.get("current_step", 0)

        updates = {
            "messages": [HumanMessage(content=result.content, name="orchestrator")],
            "last_reason": reason,
        }

        updates["current_step"] = step + 1
        return Command(update=updates, goto=goto)


     #Chart generator agent and node
    # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
    chart_agent = create_react_agent(
        llm,
        [python_repl_tool],
        prompt=make_system_prompt(
            "You are a Chart Generator. You can only generate charts using simple Python. Try to avoid importing external libraries. You are working with a researcher colleague."
        ),
    )

    @instrument(
        span_type="CHART_GENERATOR_NODE",
        attributes=lambda ret, exception, *args, **kwargs: {
            f"{BASE_SCOPE}.execution_trace": args[0].get("execution_trace", {}),
            f"{BASE_SCOPE}.user_query": args[0].get("user_query", {}),
            f"{BASE_SCOPE}.current_step": args[0].get("current_step", 0),
            f"{BASE_SCOPE}.last_reason": args[0].get("last_reason", ""),
            f"{BASE_SCOPE}.chart_generator_response": ret.update["messages"][-1].content
        },
    )
    def chart_node(state: State) -> Command[Literal["chart_eval"]]:
        with st.spinner("Chart Generator step..."):
            result = chart_agent.invoke(state)
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="chart_generator"
        )
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        chart_log = {
            "step": state.get("current_step"),
            "code": result["messages"][-2].content if len(result["messages"]) >= 2 else "",
            "output": result["messages"][-1].content,
        }
        return Command(
            update={
                "messages": result["messages"],
                "execution_trace": append_to_step_trace(state, state.get("current_step"), chart_log)
            },
            goto="chart_eval"
        )


    workflow = StateGraph(State)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("chart_eval", chart_eval_node)
    workflow.add_node("traj_eval", traj_eval_node)

    workflow.add_edge(START, "orchestrator")


    graph = workflow.compile()

    return graph


class MultiAgentWorkflow:
    def __init__(self, search_max_results: int = 5, llm_model: str = "gpt-4o", reasoning_model: str = "o1"):
        self.graph = build_graph(search_max_results=search_max_results, llm_model=llm_model, reasoning_model=reasoning_model)

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def invoke_agent_graph(self, query: str) -> Iterator[dict[str, Any] | Any]:
        return self.graph.stream(
            {
                "messages": [("user", query)],
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": 150},
            stream_mode="updates"
        )
