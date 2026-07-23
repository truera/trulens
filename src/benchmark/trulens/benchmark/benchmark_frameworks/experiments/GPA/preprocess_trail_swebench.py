"""
Preprocess SWE-Bench trace files into readable text format.

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
agent_histories: dict[str | int, list[dict]] = {}
agent_last_message_index: dict[str | int, int] = {}


def reset_state() -> None:
    """Reset global state between trace file processing."""
    global unique_agents, agent_histories, agent_last_message_index
    unique_agents.clear()
    agent_histories.clear()
    agent_last_message_index.clear()


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
        spans_to_ignore = ["process_item", "create_agent", "CodeAgent.run"]

        # Write to file
        if self.span_name not in spans_to_ignore:
            with open(output_file, "a") as f:
                f.write(
                    f"{indent_str} {self.span_name} - {self.span_id}{parent_info}\n"
                )

        # Display LLM call details
        if self.span_name == "LiteLLMModel.__call__":
            print(f"{indent_str}   Agent: {self.agent_id}")

            with open(output_file, "a") as f:
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


def extract_messages(span_attributes: dict[str, Any]) -> tuple[list[dict], int]:
    """
    Extract messages from span attributes.

    Returns:
        Tuple of (messages list, total input message count)
    """
    messages = []

    # Find all input message indices
    input_message_indices = []
    for attr_name in span_attributes:
        if attr_name.startswith("llm.input_messages.") and attr_name.endswith(
            ".message.role"
        ):
            index_str = attr_name.split(".")[2]
            try:
                input_message_indices.append(int(index_str))
            except ValueError:
                continue

    input_message_indices.sort()

    # Extract messages using found indices
    for i in input_message_indices:
        role_key = f"llm.input_messages.{i}.message.role"
        content_key = f"llm.input_messages.{i}.message.content"

        if role_key in span_attributes:
            role = span_attributes[role_key]
            content_str = span_attributes.get(content_key, "")

            if role == "tool-call":
                content_str = content_str.replace('"', "'")

            messages.append({"role": role, "content": content_str})

    total_input_messages = (
        max(input_message_indices) + 1 if input_message_indices else 0
    )

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

    return messages, total_input_messages


def find_new_messages(
    agent_id: str | int,
    current_messages: list[dict],
    total_input_messages: int,
) -> list[dict]:
    """Find new messages by comparing message indices."""
    is_first_time = agent_id not in agent_last_message_index

    if is_first_time:
        # First time seeing this agent
        agent_last_message_index[agent_id] = total_input_messages
        agent_histories[agent_id] = current_messages.copy()
        return current_messages

    last_seen_index = agent_last_message_index[agent_id]
    new_input_message_count = total_input_messages - last_seen_index

    if new_input_message_count <= 0:
        # Check for output-only messages
        if total_input_messages == 0 and len(current_messages) > 0:
            new_messages = current_messages
        else:
            new_messages = []
    else:
        # Return the last N messages
        new_messages = current_messages[-new_input_message_count:]

    # Update tracking
    if new_input_message_count > 0:
        agent_last_message_index[agent_id] = total_input_messages

    agent_histories[agent_id] = current_messages.copy()
    return new_messages


# ============================================================
# SPAN TREE BUILDING
# ============================================================


def extract_spans_from_json(
    span_data: dict[str, Any], all_spans: dict[str, Span]
) -> Span:
    """Recursively extract spans from JSON data."""
    span_id = span_data.get("span_id")
    parent_span_id = span_data.get("parent_span_id")
    span_name = span_data.get("span_name", "Unknown")
    duration = span_data.get("duration")
    new_messages = None
    agent_id = "None"
    agent_type = "None"

    if span_name == "LiteLLMModel.__call__":
        messages, total_input_messages = extract_messages(
            span_data.get("span_attributes", {})
        )
        new_messages = find_new_messages(0, messages, total_input_messages)
        agent_id = "0"

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

    return root_spans


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Main entry point for CLI usage."""
    reset_state()

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
