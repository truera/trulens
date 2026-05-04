"""Debugging utilities for TruLens instrumentation.

These utilities help analyze and visualize trace data during development.
"""


def print_span_tree(events_df) -> None:
    """Print spans as a tree structure for debugging trace hierarchy.

    Takes a DataFrame from `session.connector.get_events()` and prints
    a visual tree showing the parent-child relationships between spans.

    Args:
        events_df: DataFrame from session.connector.get_events(record_ids=[...])

    Example:
        from trulens.core import TruSession

        session = TruSession()
        session.force_flush()
        events_df = session.connector.get_events(record_ids=[record_id])
        print_span_tree(events_df)

        # Output:
        # ├─ calculator_agent [record_root]
        #   ├─ call_llm [generation]
        #   ├─ add [tool]
        #   ├─ call_llm [generation]
    """
    spans = {}
    for _, row in events_df.iterrows():
        span_id = row["trace"].get("span_id")
        parent_id = row["record"].get("parent_id")
        name = row["record"].get("name", "unnamed")
        span_type = row["record_attributes"].get("trulens.span_type", "")
        spans[span_id] = {
            "name": name,
            "type": span_type,
            "parent_id": parent_id,
            "children": [],
        }

    # Link children to parents
    roots = []
    for span_id, span in spans.items():
        parent_id = span["parent_id"]
        if parent_id and parent_id in spans:
            spans[parent_id]["children"].append(span_id)
        else:
            roots.append(span_id)

    # Print tree
    def print_node(span_id, depth=0):
        span = spans[span_id]
        indent = "  " * depth
        type_str = f" [{span['type']}]" if span["type"] else ""
        print(f"{indent}├─ {span['name']}{type_str}")
        for child_id in span["children"]:
            print_node(child_id, depth + 1)

    for root_id in roots:
        print_node(root_id)
