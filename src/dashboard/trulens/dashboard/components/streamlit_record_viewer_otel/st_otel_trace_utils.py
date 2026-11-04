from functools import lru_cache
import random
import time
from typing import Any, Dict

import altair as alt
import pandas as pd
from st_otel_trace_constants import DURATION_BINS
from st_otel_trace_constants import DURATION_LABELS
from st_otel_trace_constants import DURATION_RANGES
from st_otel_trace_constants import MAX_DURATION
from st_otel_trace_constants import MIN_DURATION
from st_otel_trace_constants import OPERATION_HIERARCHY
from st_otel_trace_constants import OPERATIONS
from st_otel_trace_constants import STATUS_CODES
from st_otel_trace_constants import STATUS_COLORS
from st_otel_trace_constants import STATUS_WEIGHTS
import streamlit as st


def generate_random_trace() -> dict:
    existing_spans = st.session_state.trace_data.get("spans", [])
    existing_span_ids = [span["spanId"] for span in existing_spans]
    # Occasionally generate a new root span (simulate multi-root traces)
    new_root = not existing_spans or (random.random() < 0.15)
    if new_root:
        parent_id = ""
        possible_roots = [
            op for op, parents in OPERATION_HIERARCHY.items() if not parents
        ]
        operation = random.choice(possible_roots)
    else:
        # Pick a random existing span as parent
        parent_span = random.choice(existing_spans)
        parent_id = parent_span["spanId"]
        # Find valid children for this parent
        possible_children = [
            op
            for op, parents in OPERATION_HIERARCHY.items()
            if parent_span["name"] in parents
        ]
        if not possible_children:
            # fallback: allow any operation
            possible_children = OPERATIONS
        operation = random.choice(possible_children)
    # Generate a new span ID
    span_id = str(random.randint(1000, 9999))
    while span_id in existing_span_ids:
        span_id = str(random.randint(1000, 9999))
    # Duration and status
    dur_rng = DURATION_RANGES.get(operation, (MIN_DURATION, MAX_DURATION))
    duration = random.uniform(*dur_rng)
    weights = STATUS_WEIGHTS.get(operation, [0.8, 0.15, 0.05])
    status = random.choices(list(STATUS_CODES.keys()), weights=weights)[0]
    # Add more attributes/events for variety
    attributes = [
        {"key": "service.name", "value": f"service_{random.randint(1, 5)}"},
        {"key": "user.id", "value": str(random.randint(100, 999))}
        if random.random() < 0.3
        else None,
        {"key": "env", "value": random.choice(["prod", "staging", "dev"])}
        if random.random() < 0.2
        else None,
    ]
    attributes = [a for a in attributes if a]
    events = []
    if random.random() < 0.2:
        events.append({"name": "event_start", "time": int(time.time())})
    if random.random() < 0.1:
        events.append({"name": "event_error", "time": int(time.time())})
    return {
        "spanId": span_id,
        "parentSpanId": parent_id,
        "name": operation,
        "startTimeUnixNano": int(time.time() * 1e9),
        "endTimeUnixNano": int((time.time() + duration) * 1e9),
        "status": {
            "code": status,
            "message": "Error occurred" if status == "ERROR" else "",
        },
        "attributes": attributes,
        "events": events,
    }


def format_duration(duration: float) -> str:
    """Format duration in seconds to a string with 3 decimal places.

    Args:
        duration: Duration in seconds

    Returns:
        Formatted duration string with 's' suffix
    """
    return f"{duration:.3f}s"


def create_status_color_scale() -> alt.Scale:
    """Create an Altair color scale for status colors.

    Returns:
        Altair Scale object configured with status colors
    """
    return alt.Scale(
        domain=list(STATUS_COLORS.keys()), range=list(STATUS_COLORS.values())
    )


def create_duration_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration buckets to the dataframe.

    Args:
        df: DataFrame containing duration column

    Returns:
        DataFrame with added duration_bucket column
    """
    df = df.copy()
    df["duration_bucket"] = pd.cut(
        df["duration"], bins=DURATION_BINS, labels=DURATION_LABELS, right=False
    )
    return df


def get_span_display_name(span: Dict[str, Any]) -> str:
    """Get a formatted display name for a span.

    Args:
        span: Dictionary containing span data

    Returns:
        Formatted string with span name, ID, and duration
    """
    return f"{span['name']} [{span['span_id']}] ({format_duration(span['duration'])})"


def display_span_attributes(span: Dict[str, Any]) -> None:
    """Display span attributes in a formatted way.

    Args:
        span: Dictionary containing span data
    """
    if span["attributes"]:
        st.write("**Attributes:**")
        for attr in span["attributes"]:
            st.write(f"- {attr['key']}: {attr['value']}")


def display_span_events(span: Dict[str, Any]) -> None:
    """Display span events in a formatted way.

    Args:
        span: Dictionary containing span data
    """
    if span["events"]:
        st.write("**Events:**")
        for event in span["events"]:
            st.write(f"- {event['name']}: {event['time']}")


def get_descendant_span_ids(df, parent_id):
    """Get descendant span IDs using memoized search for large traces."""

    # Use a tuple of (span_id, parent_id) for memoization
    @lru_cache(maxsize=1024)
    def _descendants(pid):
        descendants = set()
        children = df[df["parent_id"] == pid]["span_id"].tolist()
        for child_id in children:
            descendants.add(child_id)
            descendants.update(_descendants(child_id))
        return descendants

    return _descendants(parent_id)


def validate_trace_schema(trace_data):
    """Validate the schema of trace data.

    Args:
        trace_data: Dictionary containing trace data

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(trace_data, dict):
        return False, "Trace data must be a JSON object."
    if "spans" not in trace_data or not isinstance(trace_data["spans"], list):
        return False, "Trace data must contain a 'spans' list."
    for span in trace_data["spans"]:
        if not isinstance(span, dict):
            return False, "Each span must be a JSON object."
        if "spanId" not in span or "name" not in span:
            return False, "Each span must have 'spanId' and 'name'."
    return True, ""


def flatten_span_tree(df):
    """Return a list of dicts with span info, indentation, and error status in pre-order.

    Args:
        df: DataFrame containing span data

    Returns:
        List of dictionaries with flattened span tree information
    """
    id_to_row = {row["span_id"]: row for _, row in df.iterrows()}
    children = {}
    for _, row in df.iterrows():
        children.setdefault(row["parent_id"], []).append(row["span_id"])
    result = []

    def visit(span_id, level):
        row = id_to_row[span_id]
        result.append({
            "span_id": span_id,
            "parent_id": row["parent_id"],
            "name": row["name"],
            "service": next(
                (
                    a["value"]
                    for a in row["attributes"]
                    if a["key"] == "service.name"
                ),
                "",
            ),
            "status": row["status"],
            "duration": row["duration"],
            "relative_start": row.get("relative_start", 0),
            "relative_end": row.get("relative_end", 0),
            "level": level,
            "error": row["status"] == "ERROR",
        })
        for child_id in children.get(span_id, []):
            visit(child_id, level + 1)

    # Start from roots
    for root_id in [
        row["span_id"] for _, row in df.iterrows() if not row["parent_id"]
    ]:
        visit(root_id, 0)
    return result
