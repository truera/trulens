"""
Trace compression utilities for reducing token usage in LLM feedback functions.

This module provides functionality to compress trace data while preserving
essential information like plans, key decisions, and error details.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TRACE_SIZE = 500000  # 500KB
MAX_TRACE_SIZE_TOKEN_LIMIT = 20000  # Minimum token limit for very large traces
DEFAULT_TOKEN_LIMIT = 40000  # Default target, leaves room for model output


def safe_truncate(s: str, max_len: int) -> str:
    """
    Safely truncate a string without breaking JSON structure.
    Ensures we don't cut mid-escape sequence or leave unclosed quotes.

    Args:
        s: String to truncate
        max_len: Maximum length

    Returns:
        Truncated string with "..." suffix if truncated, or original if not a string
    """
    if not isinstance(s, str) or len(s) <= max_len:
        return s

    truncated = s[:max_len]

    # Remove any trailing backslash that could break escape sequences
    while truncated.endswith("\\"):
        truncated = truncated[:-1]

    # Try to end at a safe boundary (comma, space, or closing bracket)
    for i in range(len(truncated) - 1, max(0, len(truncated) - 50), -1):
        if truncated[i] in ",} \n":
            truncated = truncated[: i + 1]
            break

    return truncated + "..."


class TraceCompressor:
    """
    Compresses trace data for LLM context windows while preserving critical information.

    Uses a provider-based approach to handle different trace formats (LangGraph, etc.)
    while maintaining plan preservation as the highest priority.
    """

    def __init__(self):
        """Initialize the trace compressor."""
        pass

    def _normalize_trace_data(self, trace_data: Any) -> Dict[str, Any]:
        """
        Convert trace data to a normalized dictionary format.

        Args:
            trace_data: Raw trace data (string or dict)

        Returns:
            Normalized trace data as dictionary

        Raises:
            ValueError: If trace data cannot be normalized
        """

        # Convert string to dict if needed
        if isinstance(trace_data, str):
            logger.debug("Converting string trace data to dict")
            try:
                data = json.loads(trace_data)
                logger.debug(
                    f"Successfully parsed JSON, result type: {type(data)}"
                )
                logger.debug(
                    f"Parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
                )
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse trace data as JSON: {e}")
                raise ValueError("Invalid JSON trace data")
        else:
            data = trace_data
            logger.debug(f"Using trace data as-is, type: {type(data)}")
            if isinstance(data, dict):
                logger.debug(f"Input dict keys: {list(data.keys())}")

        if not isinstance(data, dict):
            logger.debug(f"Trace data is not a dictionary, type: {type(data)}")
            raise ValueError("Trace data must be a dictionary")

        logger.debug(f"Normalized trace data keys: {list(data.keys())}")
        return data

    def compress_trace_with_plan_priority(
        self, trace_data: Any, target_token_limit: int = DEFAULT_TOKEN_LIMIT
    ) -> Dict[str, Any]:
        """
        Compress trace with plan preservation as highest priority.
        If context window is exceeded, compress other data more aggressively.
        """
        logger.debug("compress_trace_with_plan_priority called")
        logger.debug(f"target_token_limit: {target_token_limit}")

        try:
            data = self._normalize_trace_data(trace_data)
        except ValueError as e:
            logger.debug(f"Failed to normalize trace data: {e}")
            return {"error": str(e)}

        # Use global provider registry
        from trulens.core.utils.trace_provider import get_trace_provider

        logger.debug("Getting trace provider")
        provider = get_trace_provider(data)
        result = provider.compress_with_plan_priority(data, target_token_limit)

        return result

    def compress_trace(self, trace_data: Any) -> Dict[str, Any]:
        """
        Compress trace data using the appropriate provider.

        Args:
            trace_data: The trace data to compress

        Returns:
            Compressed trace data
        """
        logger.debug("compress_trace called")

        try:
            data = self._normalize_trace_data(trace_data)
        except ValueError as e:
            logger.debug(f"Failed to normalize trace data: {e}")
            return {"error": str(e)}

        logger.debug("Using modified trace compression with plan preservation")

        # Use global provider registry
        from trulens.core.utils.trace_provider import get_trace_provider

        provider = get_trace_provider(data)
        result = provider.compress_trace(data)

        return result

    def _basic_compression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic compression fallback."""
        compressed = {}

        # Always try to preserve plan first
        plan = self._extract_plan_from_data(data)
        if plan:
            compressed["plan"] = plan
        else:
            pass

        # Add basic trace info
        if "trace_id" in data:
            compressed["trace_id"] = data["trace_id"]

        # Add spans (compressed)
        if "spans" in data:
            compressed["spans"] = self._compress_spans_basic(data["spans"])

        # Add metadata
        if "metadata" in data:
            compressed["metadata"] = data["metadata"]

        return compressed

    def _extract_plan_from_data(self, data: Dict[str, Any]) -> Optional[Any]:
        """Extract plan from trace data using multiple strategies."""
        # Check top-level keys
        plan_keys = [
            "plan",
            "execution_plan",
            "agent_plan",
            "workflow_plan",
            "planning",
        ]
        for key in plan_keys:
            if key in data:
                return data[key]

        # Check in spans
        if "spans" in data:
            for span in data["spans"]:
                if isinstance(span, dict):
                    # Check span attributes
                    attrs = span.get("span_attributes", {})
                    if isinstance(attrs, dict):
                        for key in plan_keys:
                            if key in attrs:
                                return attrs[key]

                        # Check LangGraph state
                        for state_key in [
                            "ai.observability.graph_node.input_state",
                            "ai.observability.graph_node.output_state",
                        ]:
                            if state_key in attrs:
                                plan = self._extract_plan_from_langgraph_state(
                                    attrs[state_key]
                                )
                                if plan:
                                    return plan

                    # Check span input/output
                    for io_key in ["input", "output", "result"]:
                        if io_key in span:
                            plan = self._extract_plan_from_langgraph_state(
                                span[io_key]
                            )
                            if plan:
                                return plan

        return None

    def _extract_plan_from_langgraph_state(
        self, state_content: Any
    ) -> Optional[Any]:
        """Extract plan from LangGraph state content."""
        if isinstance(state_content, dict):
            if "execution_plan" in state_content:
                return state_content["execution_plan"]
            if (
                "update" in state_content
                and isinstance(state_content["update"], dict)
                and "execution_plan" in state_content["update"]
            ):
                return state_content["update"]["execution_plan"]

        if isinstance(state_content, str):
            # Try to parse as JSON first
            try:
                parsed_content = json.loads(state_content)
                if isinstance(parsed_content, dict):
                    if "execution_plan" in parsed_content:
                        return parsed_content["execution_plan"]
                    if (
                        "update" in parsed_content
                        and isinstance(parsed_content["update"], dict)
                        and "execution_plan" in parsed_content["update"]
                    ):
                        return parsed_content["update"]["execution_plan"]
            except json.JSONDecodeError:
                pass

            # Check for Command string pattern or "plan" substring
            if (
                "Command(" in state_content
                and "execution_plan" in state_content
            ):
                return state_content

            # General substring search for "plan" (case-insensitive)
            if "plan" in state_content.lower() and len(state_content) > 100:
                return state_content

        return None

    def _compress_spans_basic(self, spans: List[Any]) -> List[Dict[str, Any]]:
        """Basic span compression."""
        compressed_spans = []

        for span in spans[:10]:  # Limit to first 10 spans
            if isinstance(span, dict):
                compressed_span = {
                    "span_name": span.get("span_name", "unknown"),
                    "span_id": span.get("span_id", "unknown"),
                }

                # Add important attributes
                if "span_attributes" in span:
                    attrs = span.get("span_attributes", {})
                    if isinstance(attrs, dict):
                        important_attrs = {}
                        for key, value in list(attrs.items())[
                            :5
                        ]:  # Limit attributes
                            if any(
                                important in key.lower()
                                for important in ["error", "result", "output"]
                            ):
                                important_attrs[key] = str(value)[
                                    :200
                                ]  # Truncate values
                        if important_attrs:
                            compressed_span["span_attributes"] = important_attrs

            compressed_spans.append(compressed_span)

        return compressed_spans


def _convert_trulens_trace_format(trace_data: Any) -> Dict[str, Any]:
    """
    Convert TruLens trace format to the format expected by compression providers.

    TruLens traces have events with record_attributes, but our providers expect
    spans with span_attributes.
    """
    if isinstance(trace_data, str):
        try:
            parsed_data = json.loads(trace_data)
        except json.JSONDecodeError:
            return {"trace_id": "unknown", "spans": []}
    else:
        parsed_data = trace_data

    if not isinstance(parsed_data, dict):
        return {"trace_id": "unknown", "spans": []}

    # Convert TruLens events format to spans format
    converted = {
        "trace_id": parsed_data.get("trace_id", "unknown"),
        "spans": [],
    }

    # Check if this is already in the expected format
    if "spans" in parsed_data and isinstance(parsed_data["spans"], list):
        return parsed_data

    # Handle different trace formats
    if "events" in parsed_data:
        # Format 1: {"events": [...]} - array of events
        events = parsed_data["events"]

        if isinstance(events, list):
            for i, event in enumerate(events):
                if isinstance(event, dict) and "record" in event:
                    record = event["record"]
                    if isinstance(record, dict):
                        # Convert record to span format
                        span = {
                            "span_id": str(event.get("event_id", f"span_{i}")),
                            "span_name": record.get("name", "unknown"),
                            "span_attributes": record.get(
                                "record_attributes", {}
                            ),
                        }

                        # Add parent span ID if available
                        if "parent_span_id" in record:
                            span["parent_span_id"] = str(
                                record["parent_span_id"]
                            )

                        converted["spans"].append(span)

    elif "record" in parsed_data:
        record = parsed_data["record"]
        if isinstance(record, dict):
            record_keys = list(record.keys())
            # Check if this is an array-like structure with numeric keys
            if all(
                key.isdigit() for key in record_keys[:10]
            ):  # Check first 10 keys
                # Convert each numeric entry to a span
                for i, key in enumerate(record_keys):
                    if key.isdigit():
                        event_data = record[key]
                        if isinstance(event_data, dict):
                            # Check if this is a direct record (has 'name', 'kind', etc.)
                            if "name" in event_data:
                                # This is a direct record object, but we need to get the attributes from the outer structure
                                # The record_attributes should be in the parent parsed_data
                                record_attrs = {}

                                # First, try to get record_attributes from the same level as the record
                                if "record_attributes" in parsed_data:
                                    attrs_data = parsed_data[
                                        "record_attributes"
                                    ]

                                    if isinstance(attrs_data, dict):
                                        # Check if it's an array-like structure with numeric keys
                                        if key in attrs_data:
                                            record_attrs = attrs_data[key]
                                        # Or if it's a direct attributes dict for this event
                                        elif (
                                            len(record_keys) == 1
                                        ):  # Single event case
                                            record_attrs = attrs_data

                                # Also check if record_attributes is directly in the event_data
                                if (
                                    not record_attrs
                                    and "record_attributes" in event_data
                                ):
                                    record_attrs = event_data[
                                        "record_attributes"
                                    ]

                                span = {
                                    "span_id": f"span_{key}",
                                    "span_name": event_data.get(
                                        "name", "unknown"
                                    ),
                                    "span_attributes": record_attrs
                                    if isinstance(record_attrs, dict)
                                    else {},
                                }

                                # Add parent span ID if available
                                if "parent_span_id" in event_data:
                                    span["parent_span_id"] = str(
                                        event_data["parent_span_id"]
                                    )

                                converted["spans"].append(span)

                            elif "record" in event_data:
                                # This has nested record structure
                                actual_record = event_data["record"]
                                if isinstance(actual_record, dict):
                                    span = {
                                        "span_id": str(
                                            event_data.get(
                                                "event_id", f"span_{key}"
                                            )
                                        ),
                                        "span_name": actual_record.get(
                                            "name", "unknown"
                                        ),
                                        "span_attributes": event_data.get(
                                            "record_attributes", {}
                                        ),
                                    }

                                    # Add parent span ID if available
                                    if "parent_span_id" in actual_record:
                                        span["parent_span_id"] = str(
                                            actual_record["parent_span_id"]
                                        )

                                    converted["spans"].append(span)

            else:
                # Handle as single record object
                span = {
                    "span_id": str(parsed_data.get("event_id", "span_0")),
                    "span_name": record.get("name", "unknown"),
                    "span_attributes": parsed_data.get("record_attributes", {}),
                }

                # Add parent span ID if available
                if "parent_span_id" in record:
                    span["parent_span_id"] = str(record["parent_span_id"])

                converted["spans"].append(span)

    return converted


# Convenience function for backward compatibility
def compress_trace_for_feedback(
    trace_data: Any,
    preserve_plan: bool = True,
    target_token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> Dict[str, Any]:
    """
    Convenience function to compress trace data for feedback functions.

    Args:
        trace_data: The trace data to compress
        preserve_plan: Whether to preserve complete plan (recommended for metrics)
        target_token_limit: Target token limit for context window management

    Returns:
        Compressed trace data with plan preservation prioritized
    """

    # Check for extremely large traces and scale down token limit proportionally
    trace_size = len(str(trace_data))
    if trace_size > MAX_TRACE_SIZE:
        # Scale down proportionally based on how oversized the trace is
        scale_factor = MAX_TRACE_SIZE / trace_size
        target_token_limit = max(
            int(target_token_limit * scale_factor),
            MAX_TRACE_SIZE_TOKEN_LIMIT,  # Don't go below the minimum floor
        )

    # Convert TruLens trace format to expected format
    converted_trace = _convert_trulens_trace_format(trace_data)

    compressor = TraceCompressor()

    if preserve_plan:
        result = compressor.compress_trace_with_plan_priority(
            converted_trace, target_token_limit
        )
    else:
        result = compressor.compress_trace(converted_trace)

    return result
