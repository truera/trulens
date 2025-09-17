import argparse
import json
from typing import List, Optional, Union

unique_agents = []
agent_histories = {}

manager_agent = None
search_agents = []


def reset_state():
    global unique_agents, agent_histories, manager_agent, search_agents
    unique_agents.clear()
    agent_histories.clear()
    manager_agent = None
    search_agents.clear()


class Span:
    def __init__(
        self,
        span_id,
        parent_span_id,
        span_name,
        duration=None,
        agent_id="None",
        agent_type="None",
        new_messages=None,
    ):
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.span_name = span_name
        self.duration = duration
        self.children = []
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.new_messages = new_messages

    def add_child(self, child):
        self.children.append(child)

    def format_messages(self, messages, indent_str):
        """Format messages in a readable way"""
        if not messages:
            return "No new messages"

        formatted_lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Replace newlines with spaces for cleaner display
            content = content.replace("\n", " ").replace("\r", "")

            formatted_lines.append(
                f"{indent_str}     [{role.upper()}]: {content}"
            )

        return "\n".join(formatted_lines)

    def display(self, indent=0, output_file: Optional[str] = None):
        """Display span information with indentation for tree structure"""
        if output_file is None:
            output_file = (
                "examples/experimental/traversal.txt"  # fallback default
            )

        indent_str = "  " * indent
        parent_info = (
            f" (parent: {self.parent_span_id})"
            if self.parent_span_id
            else " (root)"
        )
        print(f"{indent_str}├─ {self.span_name} - {self.span_id}{parent_info}")
        spans_to_ignore = [
            "main",
            "get_examples_to_answer",
            "answer_single_question",
            "create_agent_hierarchy",
        ]
        if self.span_name not in spans_to_ignore:
            with open(output_file, "a") as f:
                f.write(
                    f"{indent_str} {self.span_name} - {self.span_id}{parent_info}\n"
                )
        if self.span_name == "LiteLLMModel.__call__":
            print(f"{indent_str}   Agent: {self.agent_type}")
            if self.new_messages:
                print(f"{indent_str}   New Messages:")
                formatted_messages = self.format_messages(
                    self.new_messages, indent_str
                )
                print(formatted_messages)
            else:
                print(f"{indent_str}   No new messages")

            with open(output_file, "a") as f:
                f.write(f"{indent_str}   Agent: {self.agent_type}\n")
                if self.new_messages:
                    f.write(f"{indent_str}   New Messages:\n")
                    formatted_messages = self.format_messages(
                        self.new_messages, indent_str
                    )
                    f.write(formatted_messages + "\n")
                else:
                    f.write(f"{indent_str}   No new messages\n")

        for child in self.children:
            child.display(indent + 1, output_file=output_file)


def extract_messages(span_attributes: dict) -> List[dict]:
    """Extract messages as simple dicts"""
    messages = []
    i = 0
    while f"llm.input_messages.{i}.message.role" in span_attributes:
        role = span_attributes[f"llm.input_messages.{i}.message.role"]
        content_str = span_attributes[f"llm.input_messages.{i}.message.content"]
        if (
            span_attributes[f"llm.input_messages.{i}.message.role"]
            == "tool-call"
        ):
            content_str = content_str.replace('"', "'")
        messages.append({"role": role, "content": content_str})
        i += 1

    # Handle output messages - normalize to same format
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
        # Convert arguments to string with single quotes to match target format
        args_str = str(function_message["arguments"]).replace('"', "'")
        args_str = args_str.replace("'", "'")
        formatted_content = f"Calling tools:\n[{{'id': '{tool_message['id']}', 'type': '{tool_message['type']}', 'function': {{'name': '{function_message['name']}', 'arguments': {args_str}}}}}]"

        messages.append({"role": "tool-call", "content": formatted_content})

    return messages


def find_new_messages(
    agent_id: Union[str, int], current_messages: List[dict]
) -> List[dict]:
    """Find new messages by simple list comparison"""
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
                continue
        else:
            new_messages = current_messages[i:]
            break

    # Update history
    agent_histories[agent_id] = current_messages.copy()
    return new_messages


def check_parent_agent_type(span_id: str, all_spans: dict) -> str:
    """
    Check if a LiteLLMModel's parent or grandparent is manager_agent or in search_agents.
    Returns 'manager', 'search', or 'none' based on the hierarchy.
    """
    global manager_agent, search_agents

    # Get the current span
    current_span = all_spans.get(span_id)

    # Check parent
    parent_id = current_span.parent_span_id
    if parent_id:
        if parent_id == manager_agent:
            return "Manager"
        elif parent_id in search_agents:
            return "search_agent " + str(search_agents.index(parent_id))

        # Check grandparent
        parent_span = all_spans.get(parent_id)
        if parent_span and parent_span.parent_span_id:
            grandparent_id = parent_span.parent_span_id
            if grandparent_id == manager_agent:
                return "Manager"
            elif grandparent_id in search_agents:
                return "search_agent " + str(
                    search_agents.index(grandparent_id)
                )

    return "Standalone"


def extract_spans_from_json(span_data, all_spans):
    """Recursively extract spans from JSON data"""
    global manager_agent, search_agents

    span_id = span_data.get("span_id")
    parent_span_id = span_data.get("parent_span_id")
    span_name = span_data.get("span_name", "Unknown")
    duration = span_data.get("duration")
    new_messages = None
    agent_id = "None"
    agent_type = "None"
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

    # Process child spans if they exist
    if "child_spans" in span_data:
        for child_span_data in span_data["child_spans"]:
            child_span = extract_spans_from_json(child_span_data, all_spans)
            span.add_child(child_span)

    return span


def build_span_tree(trace_data):
    """Build a tree structure from the trace data"""
    all_spans = {}
    root_spans = []

    # Process all spans in the spans array
    for span_data in trace_data["spans"]:
        span = extract_spans_from_json(span_data, all_spans)
        if span.parent_span_id is None:
            root_spans.append(span)

    manager_agent_history = []
    search_agent_histories = {}
    for span_id, span in all_spans.items():
        if span.span_name == "LiteLLMModel.__call__":
            agent_type = check_parent_agent_type(span_id, all_spans)
            span.agent_type = agent_type  # Store agent type for display
            if agent_type == "Manager":
                updated_messages = []
                for msg in span.new_messages:
                    if msg not in manager_agent_history:
                        manager_agent_history.append(msg)
                        updated_messages.append(msg)
                span.new_messages = updated_messages
            elif agent_type.startswith("search_agent"):
                if agent_type not in search_agent_histories:
                    search_agent_histories[agent_type] = []
                updated_messages = []
                for msg in span.new_messages:
                    if msg not in search_agent_histories[agent_type]:
                        search_agent_histories[agent_type].append(msg)
                        updated_messages.append(msg)
                span.new_messages = updated_messages
    return root_spans


def main():
    # Load the trace data
    parser = argparse.ArgumentParser(
        description="Traverse a trace and display the span tree"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="examples/experimental/0035f455b3ff2295167a844f04d85d34.json",
        help="Input file for the span tree",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="examples/experimental/traversal.txt",
        help="Output file for the span tree",
    )
    args = parser.parse_args()
    output_file = args.output_file
    with open(args.input_file, "r") as f:
        trace_data = json.load(f)

    with open(output_file, "w") as f:
        f.write(f"Trace ID: {trace_data['trace_id']}\n\n")

    print(f"Trace ID: {trace_data['trace_id']}")
    print("=" * 80)

    # Build the span tree
    root_spans = build_span_tree(trace_data)

    # Display the tree structure
    print("\nSPAN TREE STRUCTURE:")
    print("=" * 80)
    for root_span in root_spans:
        root_span.display(output_file=output_file)

    # Display summary
    # display_span_summary(all_spans)


if __name__ == "__main__":
    main()
