"""
Trace compression utilities for reducing token usage in LLM feedback functions.

This module provides functionality to compress trace data while preserving
essential information like plans, key decisions, and error details.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        logger.info(
            f"DEBUG: _normalize_trace_data input type: {type(trace_data)}"
        )
        logger.info(
            f"DEBUG: _normalize_trace_data input length: {len(str(trace_data))}"
        )

        # Convert string to dict if needed
        if isinstance(trace_data, str):
            logger.info("DEBUG: Converting string trace data to dict")
            try:
                data = json.loads(trace_data)
                logger.info(
                    f"DEBUG: Successfully parsed JSON, result type: {type(data)}"
                )
                logger.info(
                    f"DEBUG: Parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
                )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"DEBUG: Failed to parse trace data as JSON: {e}"
                )
                raise ValueError("Invalid JSON trace data")
        else:
            data = trace_data
            logger.info(f"DEBUG: Using trace data as-is, type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"DEBUG: Input dict keys: {list(data.keys())}")

        if not isinstance(data, dict):
            logger.warning(
                f"DEBUG: Trace data is not a dictionary, type: {type(data)}"
            )
            raise ValueError("Trace data must be a dictionary")

        logger.info(f"DEBUG: Normalized trace data keys: {list(data.keys())}")
        return data

    def compress_trace_with_plan_priority(
        self, trace_data: Any, target_token_limit: int = 100000
    ) -> Dict[str, Any]:
        """
        Compress trace with plan preservation as highest priority.
        If context window is exceeded, compress other data more aggressively.
        """
        logger.info("DEBUG: compress_trace_with_plan_priority called")
        logger.info(f"DEBUG: target_token_limit: {target_token_limit}")

        try:
            data = self._normalize_trace_data(trace_data)
        except ValueError as e:
            logger.error(f"DEBUG: Failed to normalize trace data: {e}")
            return {"error": str(e)}

        # Use global provider registry
        from trulens.core.utils.trace_provider import get_trace_provider

        logger.info("DEBUG: Getting trace provider")
        provider = get_trace_provider(data)
        logger.info(f"DEBUG: Selected provider: {type(provider).__name__}")
        logger.info(
            f"DEBUG: Provider can_handle result: {provider.can_handle(data)}"
        )

        logger.info("DEBUG: Calling provider.compress_with_plan_priority")
        result = provider.compress_with_plan_priority(data, target_token_limit)
        logger.info(
            f"DEBUG: Provider returned result with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
        )
        logger.info(
            f"DEBUG: Result size: {len(json.dumps(result, default=str))} characters"
        )

        return result

    def compress_trace(self, trace_data: Any) -> Dict[str, Any]:
        """
        Compress trace data using the appropriate provider.

        Args:
            trace_data: The trace data to compress

        Returns:
            Compressed trace data
        """
        logger.info("DEBUG: compress_trace called")

        try:
            data = self._normalize_trace_data(trace_data)
        except ValueError as e:
            logger.error(f"DEBUG: Failed to normalize trace data: {e}")
            return {"error": str(e)}

        logger.info(
            "PLAN_PRESERVATION_DEBUG: Using modified trace compression with plan preservation"
        )

        # Use global provider registry
        from trulens.core.utils.trace_provider import get_trace_provider

        logger.info("DEBUG: Getting trace provider")
        provider = get_trace_provider(data)
        logger.info(f"DEBUG: Selected provider: {type(provider).__name__}")
        logger.info(
            f"DEBUG: Provider can_handle result: {provider.can_handle(data)}"
        )

        logger.info("DEBUG: Calling provider.compress_trace")
        result = provider.compress_trace(data)
        logger.info(
            f"DEBUG: Provider returned result with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
        )
        logger.info(
            f"DEBUG: Result size: {len(json.dumps(result, default=str))} characters"
        )

        return result

    def _basic_compression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic compression fallback."""
        compressed = {}

        # Always try to preserve plan first
        plan = self._extract_plan_from_data(data)
        if plan:
            compressed["plan"] = plan
            logger.info("Plan preserved completely for metrics evaluation")
        else:
            logger.warning(
                "No plan found in trace data - this may impact metrics evaluation"
            )

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
    print("🔄 TRACE_CONVERSION_DEBUG: Converting TruLens trace format")
    print(f"🔄 TRACE_CONVERSION_DEBUG: Input type: {type(trace_data)}")
    print(f"🔄 TRACE_CONVERSION_DEBUG: Input size: {len(str(trace_data))}")
    logger.warning("TRACE_CONVERSION_DEBUG: Converting TruLens trace format")
    logger.warning(f"TRACE_CONVERSION_DEBUG: Input type: {type(trace_data)}")
    logger.warning(
        f"TRACE_CONVERSION_DEBUG: Input size: {len(str(trace_data))}"
    )

    if isinstance(trace_data, str):
        logger.warning("TRACE_CONVERSION_DEBUG: Input is string, parsing JSON")
        try:
            parsed_data = json.loads(trace_data)
            logger.warning(
                f"TRACE_CONVERSION_DEBUG: Parsed JSON successfully, type: {type(parsed_data)}"
            )
        except json.JSONDecodeError as e:
            logger.warning(
                f"TRACE_CONVERSION_DEBUG: Failed to parse trace JSON: {e}"
            )
            return {"trace_id": "unknown", "spans": []}
    else:
        parsed_data = trace_data
        logger.warning(
            "TRACE_CONVERSION_DEBUG: Input is not string, using as-is"
        )

    if not isinstance(parsed_data, dict):
        logger.warning(
            f"TRACE_CONVERSION_DEBUG: Trace data is not a dict, type: {type(parsed_data)}"
        )
        return {"trace_id": "unknown", "spans": []}

    logger.warning(
        f"TRACE_CONVERSION_DEBUG: Input keys: {list(parsed_data.keys())}"
    )

    # Convert TruLens events format to spans format
    converted = {
        "trace_id": parsed_data.get("trace_id", "unknown"),
        "spans": [],
    }

    # Check if this is already in the expected format
    if "spans" in parsed_data and isinstance(parsed_data["spans"], list):
        logger.warning(
            f"TRACE_CONVERSION_DEBUG: Already in spans format, {len(parsed_data['spans'])} spans"
        )
        return parsed_data

    # Handle different trace formats
    if "events" in parsed_data:
        # Format 1: {"events": [...]} - array of events
        events = parsed_data["events"]
        print(
            f"🔄 TRACE_CONVERSION_DEBUG: Found events key, type: {type(events)}"
        )
        logger.warning(
            f"TRACE_CONVERSION_DEBUG: Found events key, type: {type(events)}"
        )
        logger.warning(
            f"TRACE_CONVERSION_DEBUG: Events count: {len(events) if isinstance(events, list) else 'non-list'}"
        )

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

                        if i < 3:  # Log first 3 spans for debugging
                            print(
                                f"🔄 TRACE_CONVERSION_DEBUG: Converted span {i}: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                            )
                            logger.warning(
                                f"TRACE_CONVERSION_DEBUG: Converted span {i}: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                            )

                if i >= 5:  # Limit debug output
                    logger.warning(
                        f"TRACE_CONVERSION_DEBUG: Stopping debug at event {i}, processing remaining silently"
                    )
                    break
    elif "record" in parsed_data:
        # Format 2: Record contains array-like structure with numeric keys
        print("🔄 TRACE_CONVERSION_DEBUG: Found record format")
        logger.warning("TRACE_CONVERSION_DEBUG: Found record format")
        print(
            f"🔄 TRACE_CONVERSION_DEBUG: Top-level parsed_data keys: {list(parsed_data.keys())}"
        )
        print(
            f"🔄 TRACE_CONVERSION_DEBUG: record_attributes present: {'record_attributes' in parsed_data}"
        )
        if "record_attributes" in parsed_data:
            attrs_data = parsed_data["record_attributes"]
            print(
                f"🔄 TRACE_CONVERSION_DEBUG: record_attributes type: {type(attrs_data)}"
            )
            if isinstance(attrs_data, dict):
                attrs_keys = list(attrs_data.keys())
                print(
                    f"🔄 TRACE_CONVERSION_DEBUG: record_attributes keys (first 10): {attrs_keys[:10]}"
                )

        record = parsed_data["record"]
        if isinstance(record, dict):
            record_keys = list(record.keys())
            print(
                f"🔄 TRACE_CONVERSION_DEBUG: Record keys (first 10): {record_keys[:10]}"
            )
            logger.warning(
                f"TRACE_CONVERSION_DEBUG: Record keys (first 10): {record_keys[:10]}"
            )

            # Check if this is an array-like structure with numeric keys
            if all(
                key.isdigit() for key in record_keys[:10]
            ):  # Check first 10 keys
                print(
                    "🔄 TRACE_CONVERSION_DEBUG: Record appears to be array-like with numeric keys"
                )
                logger.warning(
                    "TRACE_CONVERSION_DEBUG: Record appears to be array-like with numeric keys"
                )

                # Convert each numeric entry to a span
                for i, key in enumerate(record_keys):
                    if key.isdigit():
                        event_data = record[key]
                        if isinstance(event_data, dict):
                            print(
                                f"🔄 TRACE_CONVERSION_DEBUG: Processing event {key}, keys: {list(event_data.keys())[:5]}"
                            )

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
                                    print(
                                        f"🔄 TRACE_CONVERSION_DEBUG: record_attributes type: {type(attrs_data)}, keys: {list(attrs_data.keys())[:5] if isinstance(attrs_data, dict) else 'Not a dict'}"
                                    )

                                    if isinstance(attrs_data, dict):
                                        # Check if it's an array-like structure with numeric keys
                                        if key in attrs_data:
                                            record_attrs = attrs_data[key]
                                            print(
                                                f"🔄 TRACE_CONVERSION_DEBUG: Found attrs for key {key} in record_attributes array"
                                            )
                                        # Or if it's a direct attributes dict for this event
                                        elif (
                                            len(record_keys) == 1
                                        ):  # Single event case
                                            record_attrs = attrs_data
                                            print(
                                                "🔄 TRACE_CONVERSION_DEBUG: Using direct record_attributes for single event"
                                            )

                                print(
                                    f"🔄 TRACE_CONVERSION_DEBUG: Event {key} record_attrs keys: {list(record_attrs.keys())[:10] if isinstance(record_attrs, dict) else 'Not a dict'}"
                                )

                                # Also check if record_attributes is directly in the event_data
                                if (
                                    not record_attrs
                                    and "record_attributes" in event_data
                                ):
                                    record_attrs = event_data[
                                        "record_attributes"
                                    ]
                                    print(
                                        f"🔄 TRACE_CONVERSION_DEBUG: Found record_attributes in event_data for {key}"
                                    )

                                # Check for output_state specifically
                                if (
                                    isinstance(record_attrs, dict)
                                    and "ai.observability.graph_node.output_state"
                                    in record_attrs
                                ):
                                    output_state = record_attrs[
                                        "ai.observability.graph_node.output_state"
                                    ]
                                    print(
                                        f"🔄 TRACE_CONVERSION_DEBUG: Found output_state in event {key}: {str(output_state)[:100]}..."
                                    )
                                    if "Command(" in str(
                                        output_state
                                    ) and "execution_plan" in str(output_state):
                                        print(
                                            f"🔄 TRACE_CONVERSION_DEBUG: ✅ Found Command with execution_plan in event {key}!"
                                        )

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

                                if i < 3:  # Log first 3 spans for debugging
                                    attrs_keys = (
                                        list(record_attrs.keys())[:5]
                                        if isinstance(record_attrs, dict)
                                        else []
                                    )
                                    print(
                                        f"🔄 TRACE_CONVERSION_DEBUG: Converted span {key}: name='{span['span_name']}', attrs_keys={attrs_keys}"
                                    )
                                    logger.warning(
                                        f"TRACE_CONVERSION_DEBUG: Converted span {key}: name='{span['span_name']}', attrs_keys={attrs_keys}"
                                    )

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

                                    if i < 3:  # Log first 3 spans for debugging
                                        print(
                                            f"🔄 TRACE_CONVERSION_DEBUG: Converted span {key}: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                                        )
                                        logger.warning(
                                            f"TRACE_CONVERSION_DEBUG: Converted span {key}: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                                        )

                        if i >= 5:  # Limit debug output
                            print(
                                f"🔄 TRACE_CONVERSION_DEBUG: Stopping debug at event {key}, processing remaining silently"
                            )
                            break
            else:
                # Handle as single record object
                print(
                    "🔄 TRACE_CONVERSION_DEBUG: Record appears to be single object"
                )
                span = {
                    "span_id": str(parsed_data.get("event_id", "span_0")),
                    "span_name": record.get("name", "unknown"),
                    "span_attributes": parsed_data.get("record_attributes", {}),
                }

                # Add parent span ID if available
                if "parent_span_id" in record:
                    span["parent_span_id"] = str(record["parent_span_id"])

                converted["spans"].append(span)
                print(
                    f"🔄 TRACE_CONVERSION_DEBUG: Converted single span: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                )
                logger.warning(
                    f"TRACE_CONVERSION_DEBUG: Converted single span: name='{span['span_name']}', attrs_keys={list(span['span_attributes'].keys())[:5]}"
                )
        else:
            print("🔄 TRACE_CONVERSION_DEBUG: Record is not a dict")
            logger.warning("TRACE_CONVERSION_DEBUG: Record is not a dict")
    else:
        print(
            "🔄 TRACE_CONVERSION_DEBUG: No 'events' or 'record' key found in parsed data"
        )
        logger.warning(
            "TRACE_CONVERSION_DEBUG: No 'events' or 'record' key found in parsed data"
        )

        # Check for other possible structures
        for key in parsed_data.keys():
            value = parsed_data[key]
            print(
                f"🔄 TRACE_CONVERSION_DEBUG: Key '{key}' -> type: {type(value)}, size: {len(str(value))}"
            )
            logger.warning(
                f"TRACE_CONVERSION_DEBUG: Key '{key}' -> type: {type(value)}, size: {len(str(value))}"
            )

    logger.warning(
        f"TRACE_CONVERSION_DEBUG: Final conversion result - spans: {len(converted['spans'])}"
    )
    return converted


# Convenience function for backward compatibility
def compress_trace_for_feedback(
    trace_data: Any,
    preserve_plan: bool = True,
    target_token_limit: int = 100000,
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
    print("🚀 COMPRESS_DEBUG: compress_trace_for_feedback called")
    print(f"🚀 COMPRESS_DEBUG: preserve_plan: {preserve_plan}")
    print(f"🚀 COMPRESS_DEBUG: target_token_limit: {target_token_limit}")
    print(f"🚀 COMPRESS_DEBUG: trace_data type: {type(trace_data)}")
    print(f"🚀 COMPRESS_DEBUG: trace_data size: {len(str(trace_data))}")
    logger.info("DEBUG: compress_trace_for_feedback called")
    logger.info(f"DEBUG: preserve_plan: {preserve_plan}")
    logger.info(f"DEBUG: target_token_limit: {target_token_limit}")
    logger.info(f"DEBUG: trace_data type: {type(trace_data)}")
    logger.info(f"DEBUG: trace_data size: {len(str(trace_data))}")

    # Check for extremely large traces and apply aggressive pre-compression
    trace_size = len(str(trace_data))
    if trace_size > 500000:  # 500KB+ traces need aggressive handling
        print(
            f"🚀 COMPRESS_DEBUG: MASSIVE TRACE DETECTED ({trace_size} chars), applying aggressive pre-compression"
        )
        logger.warning(
            f"MASSIVE TRACE: {trace_size} characters, applying aggressive compression"
        )

        # For massive traces, use a much smaller target token limit
        target_token_limit = min(
            target_token_limit, 5000
        )  # Force very small limit
        print(
            f"🚀 COMPRESS_DEBUG: Reduced target_token_limit to {target_token_limit} for massive trace"
        )

    # Convert TruLens trace format to expected format
    converted_trace = _convert_trulens_trace_format(trace_data)
    logger.warning(
        f"TRACE_CONVERSION_DEBUG: Conversion result keys: {list(converted_trace.keys())}"
    )

    compressor = TraceCompressor()

    if preserve_plan:
        logger.info("DEBUG: Using compress_trace_with_plan_priority")
        result = compressor.compress_trace_with_plan_priority(
            converted_trace, target_token_limit
        )
    else:
        logger.info("DEBUG: Using compress_trace")
        result = compressor.compress_trace(converted_trace)

    logger.info(
        f"DEBUG: compress_trace_for_feedback returning result with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
    )
    return result
