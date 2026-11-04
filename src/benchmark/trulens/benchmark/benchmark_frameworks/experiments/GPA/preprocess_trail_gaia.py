"""
Preprocess GAIA trace files into readable text format.

This module parses JSON trace data and builds a span tree structure,
extracting agent messages and displaying them in a hierarchical format.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

# ============================================================
# GLOBAL STATE
# ============================================================

unique_agents: list[str] = []
agent_histories: dict[str, list[dict]] = {}
manager_agent: str | None = None
search_agents: list[str] = []


def reset_state() -> None:
    """Reset global state between trace file processing."""
    global unique_agents, agent_histories, manager_agent, search_agents
    unique_agents.clear()
    agent_histories.clear()
    manager_agent = None
    search_agents.clear()


# ============================================================
# SPAN CLASS
# ============================================================


class Span:
    """Represents a span in the trace tree."""

    def __init__(
        self,
        span_id: str,
        parent_span_id: str | None,
        span_name: str,
        duration: float | None = None,
        agent_id: str = "None",
        agent_type: str = "None",
        new_messages: list[dict] | None = None,
    ):
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.span_name = span_name
        self.duration = duration
        self.children: list[Span] = []
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.new_messages = new_messages

    def add_child(self, child: Span) -> None:
        """Add a child span."""
        self.children.append(child)

    def format_messages(self, messages: list[dict], indent_str: str) -> str:
        """Format messages in a readable way."""
        if not messages:
            return "No new messages"

        formatted_lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Replace newlines with spaces for cleaner display
            content = content.replace("\n", " ").replace("\r", "")
            formatted_lines.append(
                f"{indent_str}     [{role.upper()}]: {content}"
            )

        return "\n".join(formatted_lines)

    def display(self, indent: int = 0, output_file: str | None = None) -> None:
        """Display span information with indentation for tree structure."""
        if output_file is None:
            output_file = "traversal.txt"

        indent_str = "  " * indent
        parent_info = (
            f" (parent: {self.parent_span_id})"
            if self.parent_span_id
            else " (root)"
        )

        # Console output
        print(f"{indent_str}├─ {self.span_name} - {self.span_id}{parent_info}")

        # Spans to skip in file output
        spans_to_ignore = [
            "main",
            "get_examples_to_answer",
            "answer_single_question",
            "create_agent_hierarchy",
        ]

        # Write to file
        if self.span_name not in spans_to_ignore:
            with open(output_file, "a") as f:
                f.write(
                    f"{indent_str} {self.span_name} - {self.span_id}{parent_info}\n"
                )

        # Display LLM call details
        if self.span_name == "LiteLLMModel.__call__":
            print(f"{indent_str}   Agent: {self.agent_type}")

            with open(output_file, "a") as f:
                f.write(f"{indent_str}   Agent: {self.agent_type}\n")

                if self.new_messages:
                    print(f"{indent_str}   New Messages:")
                    formatted_messages = self.format_messages(
                        self.new_messages, indent_str
                    )
                    print(formatted_messages)
                    f.write(f"{indent_str}   New Messages:\n")
                    f.write(formatted_messages + "\n")
                else:
                    print(f"{indent_str}   No new messages")
                    f.write(f"{indent_str}   No new messages\n")

        # Recursively display children
        for child in self.children:
            child.display(indent + 1, output_file=output_file)


# ============================================================
# MESSAGE EXTRACTION
# ============================================================


def extract_messages(span_attributes: dict[str, Any]) -> list[dict]:
    """Extract messages from span attributes."""
    messages = []
    i = 0

    # Extract input messages
    while f"llm.input_messages.{i}.message.role" in span_attributes:
        role = span_attributes[f"llm.input_messages.{i}.message.role"]
        content_str = span_attributes[f"llm.input_messages.{i}.message.content"]

        if role == "tool-call":
            content_str = content_str.replace('"', "'")

        messages.append({"role": role, "content": content_str})
        i += 1

    # Extract output messages
    if span_attributes.get("llm.output_messages.0.message.content"):
        messages.append({
            "role": span_attributes["llm.output_messages.0.message.role"],
            "content": span_attributes["llm.output_messages.0.message.content"],
        })
    elif span_attributes.get(
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
    ):
        tool_message = json.loads(span_attributes.get("output.value", ""))[
            "tool_calls"
        ][0]
        function_message = tool_message["function"]
        args_str = str(function_message["arguments"]).replace('"', "'")

        formatted_content = (
            f"Calling tools:\n"
            f"[{{'id': '{tool_message['id']}', "
            f"'type': '{tool_message['type']}', "
            f"'function': {{'name': '{function_message['name']}', "
            f"'arguments': {args_str}}}}}]"
        )
        messages.append({"role": "tool-call", "content": formatted_content})

    return messages


def find_new_messages(
    agent_id: str, current_messages: list[dict]
) -> list[dict]:
    """Find new messages by comparing with agent's history."""
    if agent_id not in agent_histories:
        # First time seeing this agent
        agent_histories[agent_id] = current_messages.copy()
        return current_messages

    previous = agent_histories[agent_id]
    new_messages = []

    for i in range(len(previous)):
        if previous[i] == current_messages[i]:
            if i == len(previous) - 1:
                new_messages = current_messages[len(previous) :]
        else:
            new_messages = current_messages[i:]
            break

    # Update history
    agent_histories[agent_id] = current_messages.copy()
    return new_messages


# ============================================================
# AGENT TYPE DETECTION
# ============================================================


def check_parent_agent_type(span_id: str, all_spans: dict[str, Span]) -> str:
    """
    Check if a LiteLLMModel's parent or grandparent is manager_agent or search_agent.

    Returns 'Manager', 'search_agent N', or 'Standalone'.
    """
    global manager_agent, search_agents

    current_span = all_spans.get(span_id)
    if not current_span:
        return "Standalone"

    # Check parent
    parent_id = current_span.parent_span_id
    if parent_id:
        if parent_id == manager_agent:
            return "Manager"
        if parent_id in search_agents:
            return f"search_agent {search_agents.index(parent_id)}"

        # Check grandparent
        parent_span = all_spans.get(parent_id)
        if parent_span and parent_span.parent_span_id:
            grandparent_id = parent_span.parent_span_id
            if grandparent_id == manager_agent:
                return "Manager"
            if grandparent_id in search_agents:
                return f"search_agent {search_agents.index(grandparent_id)}"

    return "Standalone"


# ============================================================
# SPAN TREE BUILDING
# ============================================================


def extract_spans_from_json(
    span_data: dict[str, Any], all_spans: dict[str, Span]
) -> Span:
    """Recursively extract spans from JSON data."""
    global manager_agent, search_agents

    span_id = span_data.get("span_id")
    parent_span_id = span_data.get("parent_span_id")
    span_name = span_data.get("span_name", "Unknown")
    duration = span_data.get("duration")
    new_messages = None
    agent_id = "None"
    agent_type = "None"

    # Identify agent types
    if span_name == "CodeAgent.run":
        manager_agent = span_id
    elif span_name == "ToolCallingAgent.run":
        search_agents.append(span_id)
    elif span_name == "LiteLLMModel.__call__":
        description = span_data.get("span_attributes", {}).get(
            "llm.input_messages.0.message.content", ""
        )
        messages = extract_messages(span_data.get("span_attributes", {}))

        if description not in unique_agents:
            unique_agents.append(description)

        agent_id = str(unique_agents.index(description))
        new_messages = find_new_messages(agent_id, messages)

    span = Span(
        span_id,
        parent_span_id,
        span_name,
        duration,
        str(agent_id),
        agent_type,
        new_messages,
    )
    all_spans[span_id] = span

    # Process child spans
    if "child_spans" in span_data:
        for child_span_data in span_data["child_spans"]:
            child_span = extract_spans_from_json(child_span_data, all_spans)
            span.add_child(child_span)

    return span


def build_span_tree(trace_data: dict[str, Any]) -> list[Span]:
    """Build a tree structure from the trace data."""
    all_spans: dict[str, Span] = {}
    root_spans: list[Span] = []

    # Process all spans
    for span_data in trace_data["spans"]:
        span = extract_spans_from_json(span_data, all_spans)
        if span.parent_span_id is None:
            root_spans.append(span)

    # Track message history per agent type
    manager_agent_history: list[dict] = []
    search_agent_histories: dict[str, list[dict]] = {}

    for span_id, span in all_spans.items():
        if span.span_name == "LiteLLMModel.__call__":
            agent_type = check_parent_agent_type(span_id, all_spans)
            span.agent_type = agent_type

            # Deduplicate messages based on agent type
            if agent_type == "Manager" and span.new_messages:
                updated_messages = [
                    msg
                    for msg in span.new_messages
                    if msg not in manager_agent_history
                ]
                manager_agent_history.extend(updated_messages)
                span.new_messages = updated_messages

            elif agent_type.startswith("search_agent") and span.new_messages:
                if agent_type not in search_agent_histories:
                    search_agent_histories[agent_type] = []
                updated_messages = [
                    msg
                    for msg in span.new_messages
                    if msg not in search_agent_histories[agent_type]
                ]
                search_agent_histories[agent_type].extend(updated_messages)
                span.new_messages = updated_messages

    return root_spans


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Traverse a trace and display the span tree"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="trace.json",
        help="Input JSON file containing trace data",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="traversal.txt",
        help="Output file for the span tree",
    )
    args = parser.parse_args()

    # Load trace data
    with open(args.input_file) as f:
        trace_data = json.load(f)

    # Initialize output file
    with open(args.output_file, "w") as f:
        f.write(f"Trace ID: {trace_data['trace_id']}\n\n")

    print(f"Trace ID: {trace_data['trace_id']}")
    print("=" * 80)

    # Build and display the span tree
    root_spans = build_span_tree(trace_data)

    print("\nSPAN TREE STRUCTURE:")
    print("=" * 80)
    for root_span in root_spans:
        root_span.display(output_file=args.output_file)


if __name__ == "__main__":
    main()
