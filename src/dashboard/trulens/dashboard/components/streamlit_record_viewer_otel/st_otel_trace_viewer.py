import json
from typing import Any, Dict, Optional, Union

import altair as alt
import pandas as pd
from snowflake import telemetry
from st_otel_trace_constants import DURATION_LABELS
from st_otel_trace_constants import STATUS_CODES
from st_otel_trace_constants import STATUS_COLORS
from st_otel_trace_constants import STYLE_BLOCK_BG
from st_otel_trace_constants import STYLE_BLOCK_BORDER_RADIUS
from st_otel_trace_constants import STYLE_BLOCK_MARGIN
from st_otel_trace_constants import STYLE_BLOCK_PADDING
from st_otel_trace_constants import STYLE_DURATION_COLOR
from st_otel_trace_constants import STYLE_NAME_FONT_SIZE
from st_otel_trace_constants import STYLE_NAME_FONT_WEIGHT
from st_otel_trace_constants import STYLE_STATUS_FONT_SIZE
from st_otel_trace_utils import create_duration_buckets
from st_otel_trace_utils import create_status_color_scale
from st_otel_trace_utils import display_span_attributes
from st_otel_trace_utils import display_span_events
from st_otel_trace_utils import flatten_span_tree
from st_otel_trace_utils import format_duration
from st_otel_trace_utils import generate_random_trace
from st_otel_trace_utils import get_span_display_name
from st_otel_trace_utils import validate_trace_schema
import streamlit as st

# Page Configuration
st.set_page_config(layout="wide", page_title="OpenTelemetry Trace Viewer")


def parse_trace_data(
    trace_data: Union[str, Dict[str, Any]],
) -> Optional[pd.DataFrame]:
    """Parse OpenTelemetry trace data into a structured format.

    Args:
        trace_data: JSON string or dictionary containing trace data

    Returns:
        DataFrame containing parsed span data or None if parsing fails
    """
    try:
        if isinstance(trace_data, str):
            trace_data = json.loads(trace_data)

        spans = []
        for span in trace_data.get("spans", []):
            status_code = span.get("status", {}).get("code", "UNSET")
            if status_code not in STATUS_CODES:
                status_code = "UNSET"

            span_data = {
                "span_id": span.get("spanId", ""),
                "parent_id": span.get("parentSpanId", ""),
                "name": span.get("name", ""),
                "start_time": span.get("startTimeUnixNano", 0),
                "end_time": span.get("endTimeUnixNano", 0),
                "duration": (
                    span.get("endTimeUnixNano", 0)
                    - span.get("startTimeUnixNano", 0)
                )
                / 1e9,
                "status": status_code,
                "status_message": span.get("status", {}).get("message", ""),
                "attributes": span.get("attributes", []),
                "events": span.get("events", []),
            }
            spans.append(span_data)

        return pd.DataFrame(spans)
    except Exception as e:
        st.error(f"Error parsing trace data: {str(e)}")
        return None


def visualize_trace_timeline(df):
    """Create a timeline visualization using Altair."""
    if df is None or df.empty:
        return

    # Convert timestamps to datetime
    df["start_time"] = pd.to_datetime(df["start_time"] / 1e9, unit="s")
    df["end_time"] = pd.to_datetime(df["end_time"] / 1e9, unit="s")

    # Create a new DataFrame for the timeline visualization
    timeline_df = df.copy()
    timeline_df["duration_seconds"] = (
        timeline_df["end_time"] - timeline_df["start_time"]
    ).dt.total_seconds()

    # Format times for tooltip
    timeline_df["start_time_formatted"] = timeline_df["start_time"].dt.strftime(
        "%H:%M:%S.%f"
    )[:-3]
    timeline_df["end_time_formatted"] = timeline_df["end_time"].dt.strftime(
        "%H:%M:%S.%f"
    )[:-3]

    # Sort by start time to match waterfall view
    timeline_df = timeline_df.sort_values("start_time")

    # Create the timeline chart with OpenTelemetry colors
    chart = (
        alt.Chart(timeline_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "start_time:T",
                title="Time",
                axis=alt.Axis(format="%H:%M:%S.%L"),
            ),  # Format as HH:MM:SS.mmm
            x2="end_time:T",
            y=alt.Y(
                "name:N", title="Span Name", sort=None
            ),  # Disable default sorting to maintain our custom order
            color=alt.Color(
                "status:N", title="Status", scale=create_status_color_scale()
            ),
            tooltip=[
                "name",
                "status",
                "status_message",
                alt.Tooltip("start_time_formatted", title="Start Time"),
                alt.Tooltip("end_time_formatted", title="End Time"),
                alt.Tooltip(
                    "duration_seconds", title="Duration (s)", format=".3f"
                ),
            ],
        )
        .properties(
            width="container",
            height=alt.Step(30),  # Fixed height for each bar
        )
    )

    st.subheader("Trace Timeline")
    st.altair_chart(chart, use_container_width=True)


def display_span_details(df):
    """Display detailed information about spans."""
    if df is None or df.empty:
        return

    st.subheader("Span Details")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Span Information**")
        # Create a copy of the dataframe for display
        display_df = df[
            [
                "name",
                "span_id",
                "parent_id",
                "duration",
                "status",
                "status_message",
            ]
        ].copy()
        display_df["duration"] = display_df["duration"].apply(format_duration)

        # Initialize session state for selected rows if not exists
        if "selected_rows" not in st.session_state:
            st.session_state.selected_rows = set()

        # Add a selection column
        display_df["selected"] = False

        # Display the dataframe with selection
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Select", help="Select spans to view details", default=False
                ),
                "name": st.column_config.TextColumn("Name", help="Span name"),
                "span_id": st.column_config.TextColumn(
                    "Span ID", help="Unique identifier for the span"
                ),
                "parent_id": st.column_config.TextColumn(
                    "Parent ID", help="ID of the parent span"
                ),
                "duration": st.column_config.TextColumn(
                    "Duration", help="Duration of the span"
                ),
                "status": st.column_config.TextColumn(
                    "Status", help="Status of the span"
                ),
                "status_message": st.column_config.TextColumn(
                    "Status Message", help="Additional status information"
                ),
            },
            disabled=[
                "name",
                "span_id",
                "parent_id",
                "duration",
                "status",
                "status_message",
            ],
            key="span_selector",
            on_change=lambda: setattr(
                st.session_state, "active_tab", "Timeline & Details"
            ),
        )

        # Update selected rows in session state
        selected_indices = edited_df[edited_df["selected"]].index
        st.session_state.selected_rows = set(selected_indices)

    with col2:
        # Create a container for the details box
        details_container = st.container()
        with details_container:
            st.write("**Selected Span Details**")

            if st.session_state.selected_rows:
                # Create a scrollable container for expanders
                st.markdown(
                    """
                    <style>
                    .stExpander {
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        margin-bottom: 10px;
                    }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                # Create an expander for each selected span
                for idx in sorted(st.session_state.selected_rows):
                    selected_span = df.iloc[idx]
                    with st.expander(
                        get_span_display_name(selected_span), expanded=False
                    ):
                        st.write(f"**Span ID:** {selected_span['span_id']}")
                        st.write(f"**Parent ID:** {selected_span['parent_id']}")
                        st.write(f"**Status:** {selected_span['status']}")
                        if selected_span["status_message"]:
                            st.write(
                                f"**Status Message:** {selected_span['status_message']}"
                            )

                        display_span_attributes(selected_span)
                        display_span_events(selected_span)
            else:
                st.info("Select spans from the table to view their details")

    # Duration distribution below both columns
    st.write("---")  # Add a separator
    st.write("Duration Distribution")

    # Add duration bucket column and create chart
    df = create_duration_buckets(df)

    # Create the duration distribution chart
    duration_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "duration_bucket:N",
                title="Duration Range",
                sort=DURATION_LABELS,
            ),
            y=alt.Y("count()", title="Number of Spans"),
            color=alt.Color("status:N", scale=create_status_color_scale()),
            tooltip=["count()", "status", "duration_bucket"],
        )
        .properties(width="container")
    )
    st.altair_chart(duration_chart, use_container_width=True)


def display_span_tree(df):
    """Display the span hierarchy as a tree structure."""
    if df is None or df.empty:
        return

    st.subheader("Span Hierarchy")

    # Create a dictionary to store parent-child relationships
    tree = {}
    root_spans = set()

    # First pass: identify root spans and build parent-child relationships
    for _, row in df.iterrows():
        if not row["parent_id"]:  # This is a root span
            root_spans.add(row["span_id"])
        if row["parent_id"] not in tree:
            tree[row["parent_id"]] = []
        tree[row["parent_id"]].append((
            row["name"],
            row["status"],
            row["span_id"],
        ))

    # Display the tree structure using Streamlit's markdown with status colors
    def display_node(node_id, level=0):
        if node_id in tree:
            for child_name, child_status, child_id in tree[node_id]:
                color = STATUS_COLORS.get(child_status, "#000000")
                # Create indentation with proper spacing and tree lines
                indent = "&nbsp;" * (level * 4)
                tree_line = "└─ " if level > 0 else ""
                st.markdown(
                    f"<div style='margin-left: {level * 20}px; padding: 2px 0;'>"
                    f"<span style='color:{color}'>{indent}{tree_line}{child_name}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                display_node(child_id, level + 1)

    # Display root spans
    for root_id in root_spans:
        root = df[df["span_id"] == root_id].iloc[0]
        color = STATUS_COLORS.get(root["status"], "#000000")
        st.markdown(
            f"<div style='padding: 2px 0;'>"
            f"<span style='color:{color}'>{root['name']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        display_node(root_id)


def display_waterfall_view(df):
    """Display a collapsible tree view of the trace spans using Streamlit expanders and columns for indentation, with clear distinction between parent and child spans."""
    if df is None or df.empty:
        return

    st.subheader("Waterfall View (Tree)")

    # Build parent-child relationships
    children = {}
    for _, row in df.iterrows():
        parent_id = row["parent_id"]
        if parent_id not in children:
            children[parent_id] = []
        children[parent_id].append(row)

    def render_span(span, level=0):
        display_name = f"{span['name']} [{span['span_id']}]"
        status = span["status"]
        duration = format_duration(span["duration"])
        color = STATUS_COLORS.get(status, "#000000")
        indent_cols = [0.02] * level + [0.02] + [1]
        cols = st.columns(indent_cols)
        with cols[-2]:
            if level > 0:
                st.markdown(
                    f"<div style='border-left: 4px solid {color}; height: 100%; margin-left: 0px;'>&nbsp;</div>",
                    unsafe_allow_html=True,
                )
        with cols[-1]:
            marker = (
                f"<span style='color:#888;'>{'└─ ' if level > 0 else ''}</span>"
            )
            bg = (
                STYLE_BLOCK_BG["child"] if level > 0 else STYLE_BLOCK_BG["root"]
            )
            status_msg_html = ""
            if span["status_message"]:
                status_msg_html = f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Status Message:</b> {span['status_message']}</span>"
            # Attributes HTML
            attributes_html = ""
            if span["attributes"]:
                attributes_html = (
                    f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Attributes:</b></span><ul style='margin:0 0 0 16px; font-size:{STYLE_STATUS_FONT_SIZE};'>"
                    + "".join(
                        f"<li>{attr['key']}: {attr['value']}</li>"
                        for attr in span["attributes"]
                    )
                    + "</ul>"
                )
            # Events HTML
            events_html = ""
            if span["events"]:
                events_html = (
                    f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Events:</b></span><ul style='margin:0 0 0 16px; font-size:{STYLE_STATUS_FONT_SIZE};'>"
                    + "".join(
                        f"<li>{event['name']}: {event.get('time', '')}</li>"
                        for event in span["events"]
                    )
                    + "</ul>"
                )
            st.markdown(
                f"<div style='background:{bg}; padding:{STYLE_BLOCK_PADDING}; border-radius:{STYLE_BLOCK_BORDER_RADIUS}; margin-bottom:{STYLE_BLOCK_MARGIN};'>"
                f"{marker}<span style='color:{color}; font-weight:{STYLE_NAME_FONT_WEIGHT}; font-size:{STYLE_NAME_FONT_SIZE}'>{display_name}</span> "
                f"<span style='color:{STYLE_DURATION_COLOR}'>({duration})</span>"
                f"<br><span style='font-size: {STYLE_STATUS_FONT_SIZE};'><b>Status:</b> {status}</span>"
                + status_msg_html
                + attributes_html
                + events_html
                + "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        for child in children.get(span["span_id"], []):
            render_span(child, level + 1)

    for _, root in df[df["parent_id"] == ""].iterrows():
        display_name = f"{root['name']} [{root['span_id']}]"
        status = root["status"]
        duration = format_duration(root["duration"])
        color = STATUS_COLORS.get(status, "#000000")
        header_label = f"{display_name} ({duration})"
        status_msg_html = ""
        if root["status_message"]:
            status_msg_html = f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Status Message:</b> {root['status_message']}</span>"
        attributes_html = ""
        if root["attributes"]:
            attributes_html = (
                f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Attributes:</b></span><ul style='margin:0 0 0 16px; font-size:{STYLE_STATUS_FONT_SIZE};'>"
                + "".join(
                    f"<li>{attr['key']}: {attr['value']}</li>"
                    for attr in root["attributes"]
                )
                + "</ul>"
            )
        events_html = ""
        if root["events"]:
            events_html = (
                f"<br><span style='font-size:{STYLE_STATUS_FONT_SIZE};'><b>Events:</b></span><ul style='margin:0 0 0 16px; font-size:{STYLE_STATUS_FONT_SIZE};'>"
                + "".join(
                    f"<li>{event['name']}: {event.get('time', '')}</li>"
                    for event in root["events"]
                )
                + "</ul>"
            )
        with st.expander(label=header_label, expanded=False):
            # Move all root info into a styled div, just like children
            st.markdown(
                f"<div style='background:{STYLE_BLOCK_BG['child']}; padding:{STYLE_BLOCK_PADDING}; border-radius:{STYLE_BLOCK_BORDER_RADIUS}; margin-bottom:{STYLE_BLOCK_MARGIN};'>"
                f"<span style='color:{color}; font-weight:{STYLE_NAME_FONT_WEIGHT}; font-size:{STYLE_NAME_FONT_SIZE}'>{display_name}</span> "
                f"<span style='color:{STYLE_DURATION_COLOR}'>({duration})</span>"
                f"<br><span style='font-size: {STYLE_STATUS_FONT_SIZE};'><b>Status:</b> {status}</span>"
                + status_msg_html
                + attributes_html
                + events_html
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='height: 4px;'></div>", unsafe_allow_html=True
            )
            for child in children.get(root["span_id"], []):
                render_span(child, level=1)

            st.markdown("</div></div>", unsafe_allow_html=True)


def display_waterfall_native_bars(df):
    if df is None or df.empty:
        return

    df = df.sort_values("start_time")
    # Find root spans
    root_spans = df[df["parent_id"] == ""]
    if root_spans.empty:
        st.info("No root spans found.")
        return

    # Timeline width scaling: 1s = 400px, clamp between 400px and 2000px
    px_per_sec = 400
    min_timeline_px = 400
    max_timeline_px = 2000
    label_col_width = 350

    for _, root in root_spans.iterrows():
        # Get all descendants of this root (including root)
        def collect_subtree(span_id, acc):
            acc.append(span_id)
            children = df[df["parent_id"] == span_id]["span_id"].tolist()
            for child_id in children:
                collect_subtree(child_id, acc)

        subtree_ids = []
        collect_subtree(root["span_id"], subtree_ids)
        subtree_df = df[df["span_id"].isin(subtree_ids)].copy()
        # Calculate global timeline: earliest start and latest end in subtree
        global_start = subtree_df["start_time"].min()
        global_end = subtree_df["end_time"].max()
        subtree_df["relative_start"] = (
            subtree_df["start_time"] - global_start
        ).dt.total_seconds()
        subtree_df["relative_end"] = (
            subtree_df["end_time"] - global_start
        ).dt.total_seconds()
        total_duration = (global_end - global_start).total_seconds()
        timeline_width_px = int(total_duration * px_per_sec)
        timeline_width_px = max(
            min_timeline_px, min(timeline_width_px, max_timeline_px)
        )
        flat = flatten_span_tree(subtree_df)
        with st.expander(
            f"{root['name']} [{root['span_id']}] (Root Timeline)", expanded=True
        ):
            # Table header
            st.markdown(
                f"<div style='display: flex; font-weight: bold; margin-bottom: 8px;'>"
                f"<div style='width: {label_col_width}px;'>Span</div>"
                f"<div style='flex: 1;'>Timeline</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            min_width_px = 8
            bar_height = 36
            bar_font_size = 16
            # Timeline: horizontally scrollable, each row is a flex row with label and bar
            st.markdown(
                f"""
            <div style='overflow-x: auto; width: 100%;'>
              <div style='min-width: {timeline_width_px}px;'>
            """,
                unsafe_allow_html=True,
            )
            for row in flat:
                indent = 20 * row["level"]
                icon = "❗️" if row["error"] else ""
                border = (
                    f"border-left: 4px solid {STATUS_COLORS.get(row['status'], '#888')};"
                    if row["level"] > 0
                    else ""
                )
                bg = "background: #f5f7fa;" if row["level"] > 0 else ""
                label_html = f"<div style='width: {label_col_width}px; min-width: {label_col_width}px; max-width: {label_col_width}px; height: {bar_height}px; display: flex; align-items: center;'><span style='margin-left: {indent}px; font-weight: bold; font-size: {bar_font_size}px; {border} {bg} padding-left: 8px;'>{row['service']}: {row['name']} {icon}</span></div>"
                left_px = (
                    int(
                        (row["relative_start"] / total_duration)
                        * timeline_width_px
                    )
                    if total_duration > 0
                    else 0
                )
                width_px = (
                    int(
                        (
                            (row["relative_end"] - row["relative_start"])
                            / total_duration
                        )
                        * timeline_width_px
                    )
                    if total_duration > 0
                    else 0
                )
                is_min = width_px < min_width_px and width_px > 0
                width_px = max(width_px, min_width_px) if width_px > 0 else 0
                color = STATUS_COLORS.get(row["status"], "#888")
                bar_style = f"position: absolute; left: {left_px}px; width: {width_px}px; height: {bar_height - 6}px; background: {color}; border-radius: 6px; color: #fff; display: flex; align-items: center; font-size: {bar_font_size}px; padding-left: 12px; overflow: hidden;"
                if is_min:
                    bar_style += " border: 2px dashed #222;"
                bar_html = (
                    f"<div style='position: relative; height: {bar_height}px; width: {timeline_width_px}px; background: #f8f8f8; margin-bottom: 8px;'>"
                    f"<div title='Duration: {row['duration']:.3f}s{' (stretched for visibility)' if is_min else ''}' "
                    f"style='{bar_style}'>"
                    f"{row['duration']:.3f}s"
                    f"</div></div>"
                )
                st.markdown(
                    f"<div style='display: flex; flex-direction: row; align-items: center; margin-bottom: 0px;'>"
                    f"{label_html}"
                    f"{bar_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.divider()


def main():
    st.title("OpenTelemetry Trace Viewer")
    # Initialize session state for trace data and tab selection
    if "trace_data" not in st.session_state:
        st.session_state.trace_data = {"spans": []}
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Timeline"
    # File uploader for trace data
    uploaded_file = st.file_uploader(
        "Upload OpenTelemetry trace data (JSON)", type=["json"]
    )
    if uploaded_file is not None:
        try:
            trace_data = json.load(uploaded_file)
            valid, msg = validate_trace_schema(trace_data)
            if not valid:
                st.error(f"Invalid trace file: {msg}")
            else:
                st.session_state.trace_data = trace_data
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    # Add button to generate new trace events
    if st.button("Generate New Trace Event"):
        with st.spinner("Generating trace event..."):
            new_span = generate_random_trace()
            st.session_state.trace_data["spans"].append(new_span)
            telemetry.add_event(
                "new_trace_event",
                {
                    "span_id": new_span["spanId"],
                    "name": new_span["name"],
                    "status": new_span["status"]["code"],
                },
            )
            st.toast("New trace event generated!", icon="✅")
    # Parse and display the trace data
    df = parse_trace_data(st.session_state.trace_data)
    if df is not None:
        filtered_df = df.copy()
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Timeline & Details",
            "Hierarchy",
            "Waterfall (Tree)",
            "Waterfall (Bars)",
        ])
        if tab1:
            st.session_state.active_tab = "Timeline & Details"
        elif tab2:
            st.session_state.active_tab = "Hierarchy"
        elif tab3:
            st.session_state.active_tab = "Waterfall (Tree)"
        elif tab4:
            st.session_state.active_tab = "Waterfall (Bars)"
        with tab1:
            visualize_trace_timeline(filtered_df)
            st.divider()
            display_span_details(filtered_df)
        with tab2:
            display_span_tree(filtered_df)
        with tab3:
            display_waterfall_view(filtered_df)
        with tab4:
            display_waterfall_native_bars(filtered_df)
        with st.expander("Raw Trace Data"):
            st.json(st.session_state.trace_data)


if __name__ == "__main__":
    main()
