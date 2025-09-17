import argparse
import json
from typing import List, Optional, Tuple, Union

unique_agents = []
agent_histories = {}
agent_last_message_index = {}  # Track the last message index seen for each agent


def reset_state():
    global unique_agents, agent_histories, agent_last_message_index
    unique_agents.clear()
    agent_histories.clear()
    agent_last_message_index.clear()


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
        print(
            f"{indent_str}   DEBUG: has {len(self.children)} children, span_name={self.span_name}"
        )
        spans_to_ignore = [
            "process_item",
            "create_agent",
            "CodeAgent.run",
        ]
        if self.span_name not in spans_to_ignore:
            with open(output_file, "a") as f:
                f.write(
                    f"{indent_str} {self.span_name} - {self.span_id}{parent_info}\n"
                )
        if self.span_name == "LiteLLMModel.__call__":
            print(f"{indent_str}   Agent: {self.agent_id}")
            if self.new_messages:
                print(f"{indent_str}   New Messages:")
                formatted_messages = self.format_messages(
                    self.new_messages, indent_str
                )
                print(formatted_messages)
            else:
                print(f"{indent_str}   No new messages")
                print(
                    f"{indent_str}   DEBUG: new_messages = {self.new_messages}"
                )

            with open(output_file, "a") as f:
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


def extract_messages(span_attributes: dict) -> Tuple[List[dict], int]:
    """Extract messages as simple dicts and return total input message count"""
    messages = []

    # Find all input message indices by looking for role attributes
    input_message_indices = []
    for attr_name in span_attributes.keys():
        if attr_name.startswith("llm.input_messages.") and attr_name.endswith(
            ".message.role"
        ):
            # Extract the index from "llm.input_messages.X.message.role"
            index_str = attr_name.split(".")[2]
            try:
                index = int(index_str)
                input_message_indices.append(index)
            except ValueError:
                continue

    # Sort indices to maintain order
    input_message_indices.sort()

    # Extract messages using the found indices
    for i in input_message_indices:
        role_key = f"llm.input_messages.{i}.message.role"
        content_key = f"llm.input_messages.{i}.message.content"

        if role_key in span_attributes:
            role = span_attributes[role_key]
            content_str = span_attributes.get(content_key, "")

            if role == "tool-call":
                content_str = content_str.replace('"', "'")

            messages.append({"role": role, "content": content_str})

    # The total input message count is the highest index + 1 (since indices are 0-based)
    total_input_messages = (
        max(input_message_indices) + 1 if input_message_indices else 0
    )

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

    return messages, total_input_messages


def find_new_messages(
    agent_id: Union[str, int],
    current_messages: List[dict],
    total_input_messages: int,
) -> List[dict]:
    """Find new messages by comparing message indices instead of content"""
    is_first_time = agent_id not in agent_last_message_index

    if is_first_time:
        # First time seeing this agent
        agent_last_message_index[agent_id] = total_input_messages
        agent_histories[agent_id] = current_messages.copy()
        print(
            f"Agent {agent_id}: First time - returning all {len(current_messages)} messages (input message count: 0 -> {total_input_messages})"
        )
        return current_messages

    last_seen_index = agent_last_message_index[agent_id]

    # Calculate how many new input messages there are
    new_input_message_count = total_input_messages - last_seen_index

    if new_input_message_count <= 0:
        # No new input messages, but check if there are output messages to show
        if total_input_messages == 0 and len(current_messages) > 0:
            # Special case: this span has only output messages (like LLM responses in steps 20-23)
            new_messages = current_messages
        else:
            # No new messages at all
            new_messages = []
    else:
        # Return the last N messages where N is the number of new input messages + any output messages
        # For simplicity, just return the last new_input_message_count messages
        new_messages = current_messages[-new_input_message_count:]

    # Debug logging (optional - can be removed)
    if len(new_messages) > 0:
        print(
            f"Agent {agent_id}: Found {len(new_messages)} new messages (input message count: {last_seen_index} -> {total_input_messages})"
        )
    else:
        print(
            f"Agent {agent_id}: No new messages (input message count: {last_seen_index} -> {total_input_messages})"
        )

    # Update tracking - only update the input message index if we have new input messages
    if new_input_message_count > 0:
        agent_last_message_index[agent_id] = total_input_messages

    agent_histories[agent_id] = current_messages.copy()
    return new_messages


def extract_spans_from_json(span_data, all_spans):
    """Recursively extract spans from JSON data"""

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
        print(
            f"Span {span_id}: Found {len(messages)} messages, {total_input_messages} input messages"
        )
        new_messages = find_new_messages(0, messages, total_input_messages)
        agent_id = "0"  # Set agent_id so it shows up correctly in output
        print(
            f"Span {span_id}: new_messages = {len(new_messages) if new_messages else 0} messages"
        )

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
        print(
            f"Span {span_id} ({span_name}) has {len(span_data['child_spans'])} children"
        )
        for child_span_data in span_data["child_spans"]:
            child_span = extract_spans_from_json(child_span_data, all_spans)
            span.add_child(child_span)
            print(
                f"  Added child {child_span.span_id} ({child_span.span_name}) to {span_id}"
            )

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
            print(f"Added root span: {span.span_name} - {span.span_id}")

    print(
        f"Built tree with {len(root_spans)} root spans, {len(all_spans)} total spans"
    )
    return root_spans


def main():
    # Reset global state for each run
    reset_state()
    print("DEBUG: Global state reset")

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
    print(f"Found {len(root_spans)} root spans:")
    for i, root_span in enumerate(root_spans):
        print(f"Root span {i + 1}: {root_span.span_name} - {root_span.span_id}")
        root_span.display(output_file=output_file)

    # Display summary
    # display_span_summary(all_spans)


if __name__ == "__main__":
    main()
