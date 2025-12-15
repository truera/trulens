"""
Experimental Trace compression utilities to reduce token usage in feedback functions.
This module provides functionality to compress trace data while preserving
essential information needed for evaluation. Use with caution.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceCompressor:
    """Compresses trace data to reduce token usage while preserving essential information."""

    def __init__(self):
        """
        Initialize the trace compressor.
        """
        self.agent_histories = {}
        self.unique_agents = []

    def compress_trace(self, trace_data: Any) -> Dict[str, Any]:
        """
        Compress trace data to reduce token usage.

        Args:
            trace_data: The raw trace data to compress

        Returns:
            Compressed trace data with essential information preserved
        """
        # Convert to string if needed for processing
        if isinstance(trace_data, str):
            try:
                trace_dict = json.loads(trace_data)
            except json.JSONDecodeError:
                trace_dict = {"raw_trace": trace_data}
        elif isinstance(trace_data, dict):
            trace_dict = trace_data
        else:
            trace_dict = {"raw_trace": str(trace_data)}

        # Apply compression strategies
        compressed = self._apply_compression_strategies(trace_dict)

        return compressed

    def compress_trace_with_plan_priority(
        self, trace_data: Any, target_token_limit: int = 100000
    ) -> Dict[str, Any]:
        """
        Compress trace with plan preservation as highest priority.
        If context window is exceeded, compress other data more aggressively while keeping plan intact.

        Args:
            trace_data: The raw trace data to compress
            target_token_limit: Target token limit for context window management

        Returns:
            Compressed trace data with plan always preserved
        """
        # First, apply normal compression (which now preserves plans)
        compressed = self.compress_trace(trace_data)

        # Rough token estimation (4 chars per token approximation)
        estimated_tokens = len(json.dumps(compressed, default=str)) // 4

        if estimated_tokens <= target_token_limit:
            logger.info(
                f"Trace fits within limit: {estimated_tokens}/{target_token_limit} tokens"
            )
            return compressed

        # If still too large, compress non-plan data more aggressively
        logger.warning(
            f"Trace ({estimated_tokens} tokens) exceeds limit ({target_token_limit}), "
            f"applying aggressive compression to non-plan data while preserving plan"
        )

        # Extract and preserve plan completely
        plan = compressed.get("plan")
        plan_tokens = len(json.dumps(plan, default=str)) // 4 if plan else 0

        if plan_tokens > target_token_limit:
            logger.warning(
                f"Plan alone ({plan_tokens} tokens) exceeds target limit, but preserving for metrics"
            )

        # Rebuild with plan first, then add other data within budget
        result = {}
        if plan:
            result["plan"] = plan  # Always preserve plan completely

        # Add other data within remaining budget, prioritizing important data
        used_tokens = plan_tokens
        priority_order = [
            "execution_flow",
            "agent_interactions",
            "spans",
            "decisions",
            "issues",
            "results",
        ]

        for key in priority_order:
            if key in compressed and used_tokens < target_token_limit:
                value = compressed[key]
                value_tokens = len(json.dumps(value, default=str)) // 4

                if used_tokens + value_tokens <= target_token_limit:
                    result[key] = value
                    used_tokens += value_tokens
                else:
                    # Try to fit a truncated version for lists
                    if isinstance(value, list) and len(value) > 1:
                        for i in range(len(value) - 1, 0, -1):
                            truncated = value[:i]
                            truncated_tokens = (
                                len(json.dumps(truncated, default=str)) // 4
                            )
                            if (
                                used_tokens + truncated_tokens
                                <= target_token_limit
                            ):
                                result[key] = truncated
                                used_tokens += truncated_tokens
                                logger.info(
                                    f"Truncated {key} from {len(value)} to {len(truncated)} items"
                                )
                                break

        final_tokens = len(json.dumps(result, default=str)) // 4
        logger.info(
            f"Final compressed trace: {final_tokens} tokens (plan: {plan_tokens}, other: {final_tokens - plan_tokens})"
        )

        return result

    def _apply_compression_strategies(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply various compression strategies to the trace data.
        Preserves hierarchical structure, step names, and input/output relationships.

        Args:
            data: The trace data to compress

        Returns:
            Compressed trace data
        """
        # First, check if this is columnar format from DataFrame.to_json()
        if self._is_columnar_format(data):
            data = self._convert_columnar_to_spans(data)

        compressed = {}

        # 1. Preserve hierarchical span structure with parent/child relationships
        if "spans" in data and data["spans"]:
            compressed["spans"] = self._compress_spans_hierarchical(
                data["spans"]
            )

        # 2. Extract execution flow with step names and relationships
        execution_flow = self._extract_execution_flow_detailed(data)
        if execution_flow:
            compressed["execution_flow"] = execution_flow

        # 3. Extract agent interactions with inputs/outputs preserved
        interactions = self._extract_agent_interactions_detailed(data)
        if interactions:
            compressed["agent_interactions"] = interactions

        # 4. Extract plan if present (CRITICAL: preserve completely for metrics)
        plan_found = False
        # Check for plan in multiple common locations
        plan_locations = [
            "plan",
            "execution_plan",
            "agent_plan",
            "workflow_plan",
        ]
        for plan_key in plan_locations:
            if plan_key in data:
                compressed["plan"] = self._compress_plan(data[plan_key])
                logger.info(
                    f"Plan found in '{plan_key}' and preserved completely for metrics evaluation"
                )
                plan_found = True
                break

        # Check for LangGraph Command structure: Command(update={'execution_plan': {...}})
        if not plan_found:
            for key in ["update", "command_update", "state_update"]:
                if key in data and isinstance(data[key], dict):
                    update_data = data[key]
                    for plan_key in plan_locations:
                        if plan_key in update_data:
                            compressed["plan"] = self._compress_plan(
                                update_data[plan_key]
                            )
                            logger.info(
                                f"Plan found in {key}.{plan_key} and preserved completely for metrics evaluation"
                            )
                            plan_found = True
                            break
                    if plan_found:
                        break

        if not plan_found and "spans" in data:
            # Try to extract plan from span attributes
            for span in data["spans"]:
                if isinstance(span, dict):
                    # Check for plan in multiple locations within span
                    for plan_key in plan_locations:
                        if plan_key in span:
                            compressed["plan"] = self._compress_plan(
                                span[plan_key]
                            )
                            logger.info(
                                f"Plan found in span.{plan_key} and preserved completely for metrics evaluation"
                            )
                            plan_found = True
                            break

                    if not plan_found and "span_attributes" in span:
                        attrs = span["span_attributes"]
                        if isinstance(attrs, dict):
                            # Check direct plan keys in attributes
                            for plan_key in plan_locations:
                                if plan_key in attrs:
                                    compressed["plan"] = self._compress_plan(
                                        attrs[plan_key]
                                    )
                                    logger.info(
                                        f"Plan found in span_attributes.{plan_key} and preserved completely for metrics evaluation"
                                    )
                                    plan_found = True
                                    break

                            # Check LangGraph observability attributes for state information
                            if not plan_found:
                                langgraph_state_keys = [
                                    "ai.observability.graph_node.input_state",
                                    "ai.observability.graph_node.output_state",
                                    "ai.observability.call.kwargs.input",
                                    "ai.observability.call.return",
                                ]
                                for state_key in langgraph_state_keys:
                                    if state_key in attrs:
                                        state_value = attrs[state_key]
                                        if isinstance(state_value, str):
                                            # First try JSON parsing
                                            try:
                                                parsed_state = json.loads(
                                                    state_value
                                                )
                                                if isinstance(
                                                    parsed_state, dict
                                                ):
                                                    for (
                                                        plan_key
                                                    ) in plan_locations:
                                                        if (
                                                            plan_key
                                                            in parsed_state
                                                        ):
                                                            compressed[
                                                                "plan"
                                                            ] = self._compress_plan(
                                                                parsed_state[
                                                                    plan_key
                                                                ]
                                                            )
                                                            logger.info(
                                                                f"Plan found in span_attributes.{state_key}.{plan_key} and preserved completely for metrics evaluation"
                                                            )
                                                            plan_found = True
                                                            break
                                                    if plan_found:
                                                        break
                                            except Exception:
                                                pass

                                            # If JSON parsing failed, check for any string containing "plan"
                                            if (
                                                not plan_found
                                                and "plan"
                                                in state_value.lower()
                                            ):
                                                # This is likely a Command string representation or state containing the plan
                                                compressed["plan"] = (
                                                    self._compress_plan(
                                                        state_value
                                                    )
                                                )
                                                logger.info(
                                                    f"Plan found in span_attributes.{state_key} string (contains 'plan') and preserved completely for metrics evaluation"
                                                )
                                                plan_found = True
                                                break
                                        elif isinstance(state_value, dict):
                                            for plan_key in plan_locations:
                                                if plan_key in state_value:
                                                    compressed["plan"] = (
                                                        self._compress_plan(
                                                            state_value[
                                                                plan_key
                                                            ]
                                                        )
                                                    )
                                                    logger.info(
                                                        f"Plan found in span_attributes.{state_key}.{plan_key} and preserved completely for metrics evaluation"
                                                    )
                                                    plan_found = True
                                                    break
                                            if plan_found:
                                                break
                                    if plan_found:
                                        break

                    if plan_found:
                        break

        # Additional plan search in more locations
        if not plan_found:
            # Look for plan in processed_content or other common locations
            search_locations = [
                "processed_content",
                "events",
                "raw_trace",
                "state",
                "langgraph_state",
                "workflow_state",
            ]
            for key in search_locations:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, dict):
                                for plan_key in plan_locations:
                                    if plan_key in parsed:
                                        compressed["plan"] = (
                                            self._compress_plan(
                                                parsed[plan_key]
                                            )
                                        )
                                        logger.info(
                                            f"Plan found in {key}.{plan_key} and preserved completely for metrics evaluation"
                                        )
                                        plan_found = True
                                        break
                                if plan_found:
                                    break
                        except Exception:
                            pass
                    elif isinstance(value, dict):
                        for plan_key in plan_locations:
                            if plan_key in value:
                                compressed["plan"] = self._compress_plan(
                                    value[plan_key]
                                )
                                logger.info(
                                    f"Plan found in {key}.{plan_key} and preserved completely for metrics evaluation"
                                )
                                plan_found = True
                                break
                        if plan_found:
                            break

            # Look for plan-related fields in LangGraph/agent traces
            if not plan_found and "spans" in data:
                plan_keywords = [
                    "plan",
                    "planning",
                    "strategy",
                    "steps",
                    "workflow",
                    "agent_plan",
                    "execution_plan",
                ]
                for span in data["spans"]:
                    if isinstance(span, dict):
                        # Check span name - any node could output a plan
                        span_name = span.get("span_name", "").lower()
                        # Don't limit to specific node names - any node could create a plan

                        # Check if this span has plan-like content
                        for attr_key in [
                            "input",
                            "output",
                            "span_attributes",
                            "state",
                            "update",
                            "command",
                            "result",
                        ]:
                            if attr_key in span:
                                attr_value = span[attr_key]
                                if isinstance(attr_value, dict):
                                    # Look for execution_plan or other plan keys directly
                                    for plan_key in (
                                        plan_locations + plan_keywords
                                    ):
                                        if plan_key in attr_value:
                                            compressed["plan"] = (
                                                self._compress_plan(
                                                    attr_value[plan_key]
                                                )
                                            )
                                            logger.info(
                                                f"Plan found in span '{span_name}' {attr_key}.{plan_key} and preserved completely"
                                            )
                                            plan_found = True
                                            break

                                    # Look for LangGraph Command structure: {update: {execution_plan: {...}}}
                                    if not plan_found:
                                        for cmd_key in [
                                            "update",
                                            "command_update",
                                            "state_update",
                                        ]:
                                            if (
                                                cmd_key in attr_value
                                                and isinstance(
                                                    attr_value[cmd_key], dict
                                                )
                                            ):
                                                cmd_data = attr_value[cmd_key]
                                                for plan_key in plan_locations:
                                                    if plan_key in cmd_data:
                                                        compressed["plan"] = (
                                                            self._compress_plan(
                                                                cmd_data[
                                                                    plan_key
                                                                ]
                                                            )
                                                        )
                                                        logger.info(
                                                            f"Plan found in span '{span_name}' {attr_key}.{cmd_key}.{plan_key} and preserved completely"
                                                        )
                                                        plan_found = True
                                                        break
                                                if plan_found:
                                                    break

                                    if plan_found:
                                        break
                                elif isinstance(attr_value, str):
                                    # Try to parse JSON strings that might contain plans
                                    if (
                                        attr_value.startswith("{")
                                        and len(attr_value) > 50
                                    ):
                                        try:
                                            parsed_attr = json.loads(attr_value)
                                            if isinstance(parsed_attr, dict):
                                                # Look for plan keys directly
                                                for plan_key in (
                                                    plan_locations
                                                    + plan_keywords
                                                ):
                                                    if plan_key in parsed_attr:
                                                        compressed["plan"] = (
                                                            self._compress_plan(
                                                                parsed_attr[
                                                                    plan_key
                                                                ]
                                                            )
                                                        )
                                                        logger.info(
                                                            f"Plan found in span '{span_name}' {attr_key} JSON.{plan_key} and preserved completely"
                                                        )
                                                        plan_found = True
                                                        break

                                                # Look for LangGraph Command structure in JSON
                                                if not plan_found:
                                                    for cmd_key in [
                                                        "update",
                                                        "command_update",
                                                        "state_update",
                                                    ]:
                                                        if (
                                                            cmd_key
                                                            in parsed_attr
                                                            and isinstance(
                                                                parsed_attr[
                                                                    cmd_key
                                                                ],
                                                                dict,
                                                            )
                                                        ):
                                                            cmd_data = (
                                                                parsed_attr[
                                                                    cmd_key
                                                                ]
                                                            )
                                                            for (
                                                                plan_key
                                                            ) in plan_locations:
                                                                if (
                                                                    plan_key
                                                                    in cmd_data
                                                                ):
                                                                    compressed[
                                                                        "plan"
                                                                    ] = self._compress_plan(
                                                                        cmd_data[
                                                                            plan_key
                                                                        ]
                                                                    )
                                                                    logger.info(
                                                                        f"Plan found in span '{span_name}' {attr_key} JSON.{cmd_key}.{plan_key} and preserved completely"
                                                                    )
                                                                    plan_found = True
                                                                    break
                                                            if plan_found:
                                                                break

                                                if plan_found:
                                                    break
                                        except Exception:
                                            pass

                                    # Also check for Command(...) string patterns or any plan mention
                                    elif "plan" in attr_value.lower():
                                        # Try to extract the execution_plan from various string representations
                                        try:
                                            # Check for Command string pattern
                                            if "Command(" in attr_value and (
                                                "'update':" in attr_value
                                                or '"update":' in attr_value
                                            ):
                                                # This is a Command string representation containing the plan
                                                compressed["plan"] = (
                                                    self._compress_plan(
                                                        attr_value
                                                    )
                                                )
                                                logger.info(
                                                    f"Plan found in span '{span_name}' {attr_key} Command string and preserved completely"
                                                )
                                                plan_found = True
                                                break
                                            # Check for any string containing "plan" (could be serialized state)
                                            elif (
                                                len(attr_value) > 100
                                            ):  # Reasonable size for a plan
                                                compressed["plan"] = (
                                                    self._compress_plan(
                                                        attr_value
                                                    )
                                                )
                                                logger.info(
                                                    f"Plan found in span '{span_name}' {attr_key} string containing 'plan' and preserved completely"
                                                )
                                                plan_found = True
                                                break
                                        except Exception:
                                            pass

                                    # Check if string content looks like a plan (from any node)
                                    if len(attr_value) > 50:
                                        attr_lower = attr_value.lower()
                                        # Look for plan-like content indicators
                                        plan_indicators = [
                                            "step",
                                            "action",
                                            "tool",
                                            "execute",
                                            "agent",
                                            "plan_summary",
                                            "combination_strategy",
                                        ]
                                        if any(
                                            keyword in attr_lower
                                            for keyword in plan_indicators
                                        ):
                                            compressed["plan"] = (
                                                self._compress_plan(attr_value)
                                            )
                                            logger.info(
                                                f"Plan-like content found in span '{span_name}' {attr_key} and preserved completely"
                                            )
                                            plan_found = True
                                            break
                        if plan_found:
                            break

        if not plan_found:
            logger.warning(
                "No plan found in trace data - this may impact metrics evaluation"
            )
            # Log detailed structure to help debug
            if isinstance(data, dict):
                logger.warning(
                    f"DEBUG: Top-level trace keys: {list(data.keys())}"
                )

                # Log span structure if present
                if "spans" in data and isinstance(data["spans"], list):
                    logger.warning(f"DEBUG: Found {len(data['spans'])} spans")
                    for i, span in enumerate(
                        data["spans"][:3]
                    ):  # Log first 3 spans
                        if isinstance(span, dict):
                            span_name = span.get("span_name", "unknown")
                            span_keys = list(span.keys())
                            logger.warning(
                                f"DEBUG: Span {i} '{span_name}' keys: {span_keys}"
                            )

                            # Check span attributes
                            if "span_attributes" in span and isinstance(
                                span["span_attributes"], dict
                            ):
                                attr_keys = list(span["span_attributes"].keys())
                                logger.warning(
                                    f"DEBUG: Span {i} attributes keys: {attr_keys}"
                                )

                                # Log LangGraph state content if present
                                attrs = span["span_attributes"]
                                for state_key in [
                                    "ai.observability.graph_node.input_state",
                                    "ai.observability.graph_node.output_state",
                                ]:
                                    if state_key in attrs:
                                        state_value = attrs[state_key]
                                        if isinstance(state_value, str):
                                            logger.warning(
                                                f"DEBUG: {state_key} is string of length {len(state_value)}"
                                            )
                                            if "execution_plan" in state_value:
                                                logger.warning(
                                                    f"DEBUG: Found 'execution_plan' in {state_key}!"
                                                )
                                            # Show first 200 chars to see the structure
                                            logger.warning(
                                                f"DEBUG: {state_key} content preview: {state_value[:200]}..."
                                            )
                                        elif isinstance(state_value, dict):
                                            state_keys = list(
                                                state_value.keys()
                                            )
                                            logger.warning(
                                                f"DEBUG: {state_key} dict keys: {state_keys}"
                                            )
                                            if "execution_plan" in state_keys:
                                                logger.warning(
                                                    f"DEBUG: Found 'execution_plan' key in {state_key}!"
                                                )

                # Log other potential locations
                for key in ["processed_content", "events", "raw_trace"]:
                    if key in data:
                        value = data[key]
                        if isinstance(value, str):
                            logger.warning(
                                f"DEBUG: {key} is string of length {len(value)}"
                            )
                            if value.startswith("{"):
                                logger.warning(
                                    f"DEBUG: {key} appears to be JSON"
                                )
                        elif isinstance(value, dict):
                            logger.warning(
                                f"DEBUG: {key} dict keys: {list(value.keys())}"
                            )
                        else:
                            logger.warning(f"DEBUG: {key} type: {type(value)}")
            else:
                logger.warning(f"DEBUG: Trace data type: {type(data)}")

        # 5. Extract key decisions with context
        decisions = self._extract_key_decisions_with_context(data)
        if decisions:
            compressed["key_decisions"] = decisions

        # 6. Extract errors and warnings
        if self._has_issues(data):
            compressed["issues"] = self._extract_issues(data)

        # 7. Extract final results
        results = self._extract_results(data)
        if results:
            compressed["results"] = results

        # If compression resulted in very little data, include key raw data
        if len(compressed) < 3:
            # Include at least the essential raw structure
            compressed["trace_summary"] = self._create_trace_summary(data)

        return compressed

    def _create_trace_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the trace when compression yields too little data."""
        summary = {
            "total_spans": len(data.get("spans", [])),
            "span_names": [],
            "tools_used": [],
            "has_plan": False,
            "has_messages": False,
            "key_attributes": {},
        }

        if "spans" in data:
            for span in data["spans"]:
                if isinstance(span, dict):
                    # Collect span names
                    if "span_name" in span:
                        summary["span_names"].append(span["span_name"])

                    # Collect tools
                    if "tool" in span:
                        summary["tools_used"].append(span["tool"])

                    # Check for plan
                    if "plan" in span or (
                        "span_attributes" in span
                        and "plan" in span.get("span_attributes", {})
                    ):
                        summary["has_plan"] = True

                    # Extract key attributes
                    if "span_attributes" in span:
                        attrs = span["span_attributes"]
                        if isinstance(attrs, dict):
                            for key in [
                                "input",
                                "output",
                                "query",
                                "result",
                                "error",
                            ]:
                                if key in attrs and attrs[key]:
                                    if key not in summary["key_attributes"]:
                                        summary["key_attributes"][key] = []
                                    summary["key_attributes"][key].append(
                                        attrs[key]
                                    )

        # Include any messages
        if "messages" in data:
            summary["has_messages"] = True
            summary["message_count"] = len(data["messages"])

        return summary

    def _is_columnar_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in columnar format from DataFrame.to_json()."""
        # Columnar format has columns as keys, with row indices as nested keys
        if "trace" in data and "record" in data and "record_attributes" in data:
            # Check if the values are dicts with numeric string keys
            if isinstance(data.get("trace"), dict):
                first_key = next(iter(data["trace"].keys()), None)
                if first_key and first_key.isdigit():
                    return True
        return False

    def _convert_columnar_to_spans(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert columnar DataFrame format to span-based format."""
        spans = []

        # Get the number of rows
        num_rows = len(data.get("trace", {}))

        for i in range(num_rows):
            span = {}

            # Extract trace info
            if "trace" in data and str(i) in data["trace"]:
                trace_info = data["trace"][str(i)]
                if isinstance(trace_info, dict):
                    span["span_id"] = trace_info.get("span_id", f"span-{i}")
                    span["parent_span_id"] = trace_info.get("parent_span_id")

            # Extract record info
            if "record" in data and str(i) in data["record"]:
                record_info = data["record"][str(i)]
                if isinstance(record_info, dict):
                    span["span_name"] = record_info.get("name", f"step-{i}")
                else:
                    span["span_name"] = str(record_info)

            # Extract attributes
            if (
                "record_attributes" in data
                and str(i) in data["record_attributes"]
            ):
                attrs = data["record_attributes"][str(i)]
                if isinstance(attrs, dict):
                    span["span_attributes"] = attrs

                    # Extract common fields to top level for easier access
                    if "input" in attrs:
                        span["input"] = attrs["input"]
                    if "output" in attrs:
                        span["output"] = attrs["output"]
                    if "tool" in attrs:
                        span["tool"] = attrs["tool"]
                    if "plan" in attrs:
                        span["plan"] = attrs["plan"]

            # Extract processed content
            if (
                "processed_content" in data
                and str(i) in data["processed_content"]
            ):
                span["processed_content"] = data["processed_content"][str(i)]

            # Extract messages if present
            if "messages" in data and str(i) in data["messages"]:
                span["messages"] = data["messages"][str(i)]

            spans.append(span)

        # Return in standard format
        return {
            "spans": spans,
            "trace_id": data.get("trace_id", "unknown"),
            "metadata": data.get("metadata", {}),
        }

    def _compress_spans_hierarchical(
        self, spans: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Compress spans while preserving hierarchical parent/child relationships.

        Args:
            spans: List of span data

        Returns:
            Compressed spans with hierarchy preserved
        """
        compressed_spans = []

        for span in spans:
            if not isinstance(span, dict):
                continue

            # Skip debug/log spans unless they contain errors
            span_name = span.get("span_name", "").lower()
            if any(
                skip in span_name
                for skip in ["debug", "log", "trace", "monitor"]
            ):
                # Check if it has an error or important info
                has_error = False
                if "span_attributes" in span:
                    attrs = span.get("span_attributes", {})
                    if isinstance(attrs, dict):
                        if any(
                            key in attrs
                            for key in ["error", "exception", "failure"]
                        ):
                            has_error = True
                if not has_error:
                    continue  # Skip this debug span

            compressed_span = {
                "span_id": span.get("span_id", "unknown"),
                "parent_span_id": span.get("parent_span_id"),
                "span_name": span.get("span_name", "unknown"),
            }

            # Extract key attributes while removing verbose data
            attrs = span.get("span_attributes", {})
            if attrs:
                compressed_attrs = {}

                # Preserve important attributes with truncation
                for key, value in attrs.items():
                    if self._is_important_attribute(key):
                        # Use smaller limits for attributes
                        compressed_attrs[key] = self._compress_attribute_value(
                            key, value, max_length=150
                        )

                if compressed_attrs:
                    compressed_span["attributes"] = compressed_attrs

            # Preserve inputs and outputs with aggressive truncation
            if "input" in span:
                compressed_span["input"] = self._compress_attribute_value(
                    "input", span["input"], max_length=200
                )
            if "output" in span:
                compressed_span["output"] = self._compress_attribute_value(
                    "output", span["output"], max_length=200
                )

            # Recursively compress child spans
            if "child_spans" in span:
                compressed_span["child_spans"] = (
                    self._compress_spans_hierarchical(span["child_spans"])
                )

            compressed_spans.append(compressed_span)

        return compressed_spans

    def _is_important_attribute(self, key: str) -> bool:
        """Determine if an attribute is important enough to preserve."""
        important_patterns = [
            "plan",
            "step",
            "action",
            "decision",
            "tool",
            "function",
            "query",
            "result",
            "error",
            "input",
            "output",
            "message",
            "role",
            "content",
            "name",
            "type",
            "status",
        ]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in important_patterns)

    def _compress_attribute_value(
        self, key: str, value: Any, max_length: int = 200
    ) -> Any:
        """Compress an attribute value by truncating large content and removing redundancy."""
        # Keep simple values as-is
        if isinstance(value, (int, float, bool, type(None))):
            return value

        if isinstance(value, str):
            # Normalize whitespace first
            value = " ".join(value.split())

            # Aggressive truncation for large strings
            if len(value) > max_length:
                # Preserve slightly more for critical fields only
                if any(
                    important in key.lower()
                    for important in ["plan", "error", "decision"]
                ):
                    max_length = min(
                        max_length * 2, 400
                    )  # Max 400 chars for important fields

                if len(value) > max_length:
                    # Take beginning and end to preserve context
                    half = max_length // 2 - 20
                    value = f"{value[:half]}...[{len(value) - max_length} chars]...{value[-half:]}"

            return value

        if isinstance(value, list):
            # Limit list size for non-critical fields
            max_items = 5 if "plan" not in key.lower() else 10

            compressed = []
            seen = set()
            for i, v in enumerate(value[:max_items]):
                v_compressed = self._compress_attribute_value(
                    key, v, max_length=100
                )  # Smaller limit for list items
                v_hash = (
                    str(v_compressed)
                    if not isinstance(v_compressed, (list, dict))
                    else json.dumps(v_compressed, sort_keys=True, default=str)
                )

                if v_hash not in seen:
                    seen.add(v_hash)
                    compressed.append(v_compressed)

            if len(value) > max_items:
                compressed.append(
                    f"[{len(value) - max_items} more items omitted]"
                )
            elif len(compressed) < len(value[:max_items]):
                compressed.append(
                    f"[{len(value[:max_items]) - len(compressed)} duplicate items removed]"
                )

            return compressed

        if isinstance(value, dict):
            # Prioritize important keys
            compressed = {}
            important_keys = [
                "plan",
                "error",
                "input",
                "output",
                "query",
                "decision",
                "action",
                "result",
                "tool",
            ]
            other_keys = []

            for k in value.keys():
                if any(imp in k.lower() for imp in important_keys):
                    compressed[k] = self._compress_attribute_value(
                        f"{key}.{k}", value[k], max_length=150
                    )
                else:
                    other_keys.append(k)

            # Add a few other keys if space permits
            for k in other_keys[:3]:  # Limit to 3 additional keys
                v_compressed = self._compress_attribute_value(
                    f"{key}.{k}", value[k], max_length=50
                )
                if (
                    v_compressed is not None
                    and v_compressed != ""
                    and v_compressed != []
                    and v_compressed != {}
                ):
                    compressed[k] = v_compressed

            if len(other_keys) > 5:
                compressed["_omitted_keys"] = (
                    f"[{len(other_keys) - 5} keys omitted]"
                )

            return compressed

        return self._summarize_content(value)

    def _extract_execution_flow_detailed(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract detailed execution flow preserving step names and relationships."""
        flow = []

        # Extract from spans with hierarchy
        if "spans" in data:
            flow = self._build_flow_from_spans(data["spans"])

        # Extract from calls or steps
        elif "calls" in data:
            for i, call in enumerate(data.get("calls", [])):
                if isinstance(call, dict):
                    flow_item = {
                        "step": i + 1,
                        "name": call.get(
                            "name", call.get("function", "unknown")
                        ),
                        "type": call.get("type", "call"),
                    }

                    # Add input/output if available
                    if "input" in call:
                        flow_item["input_summary"] = self._summarize_content(
                            call["input"]
                        )
                    if "output" in call:
                        flow_item["output_summary"] = self._summarize_content(
                            call["output"]
                        )

                    flow.append(flow_item)

        return flow  # Return complete flow without limiting

    def _build_flow_from_spans(
        self, spans: List[Any], parent_id: Optional[str] = None, level: int = 0
    ) -> List[Dict[str, Any]]:
        """Build execution flow from spans preserving hierarchy."""
        flow = []

        for span in spans:
            if not isinstance(span, dict):
                continue

            # Skip debug/log spans in execution flow
            span_name = span.get("span_name", "").lower()
            if any(
                skip in span_name
                for skip in ["debug", "log", "trace", "monitor"]
            ):
                # Unless it has important info
                attrs = span.get("span_attributes", {})
                if not any(
                    key in attrs
                    for key in [
                        "error",
                        "exception",
                        "failure",
                        "plan",
                        "decision",
                    ]
                ):
                    continue

            span_parent = span.get("parent_span_id")
            if span_parent == parent_id or (
                parent_id is None and not span_parent
            ):
                flow_item = {
                    "level": level,
                    "span_id": span.get("span_id", "unknown"),
                    "parent_id": span_parent,
                    "name": span.get("span_name", "unknown"),
                }

                # Add key attributes
                attrs = span.get("span_attributes", {})
                if attrs:
                    # Extract tool/function calls
                    if "tool" in attrs or "function" in attrs:
                        flow_item["tool"] = attrs.get(
                            "tool", attrs.get("function")
                        )

                    # Extract queries or prompts
                    for key in ["query", "prompt", "question"]:
                        if key in attrs:
                            flow_item[key] = self._summarize_content(attrs[key])
                            break

                flow.append(flow_item)

                # Recursively add children
                if "child_spans" in span:
                    child_flow = self._build_flow_from_spans(
                        span["child_spans"], span.get("span_id"), level + 1
                    )
                    flow.extend(child_flow)

        return flow

    def _extract_agent_interactions_detailed(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract agent interactions preserving inputs and outputs."""
        interactions = []

        # Extract messages with full context
        if "messages" in data:
            messages = data["messages"]
            if isinstance(messages, list):
                for i, msg in enumerate(messages):  # Process all messages
                    if isinstance(msg, dict):
                        interaction = {
                            "index": i,
                            "role": msg.get("role", "unknown"),
                            "content": self._compress_message_content(
                                msg.get("content", "")
                            ),
                        }

                        # Preserve tool calls
                        if "tool_calls" in msg:
                            interaction["tool_calls"] = (
                                self._compress_tool_calls(msg["tool_calls"])
                            )

                        interactions.append(interaction)

        # Extract from spans
        if "spans" in data:
            span_interactions = self._extract_span_interactions(data["spans"])
            interactions.extend(span_interactions)

        return interactions

    def _compress_message_content(
        self, content: Any, max_length: int = 300
    ) -> Any:
        """Compress message content by truncating large content."""
        if content is None:
            return None

        # If it's already a dict or list, compress recursively
        if isinstance(content, dict):
            return self._compress_attribute_value(
                "content", content, max_length=max_length
            )
        elif isinstance(content, list):
            return self._compress_attribute_value(
                "content", content, max_length=max_length
            )

        # For strings, aggressively truncate
        content_str = str(content)

        # For very large content (likely web pages or documents), aggressively truncate
        if len(content_str) > max_length:
            # Take a small sample from beginning and end
            quarter = max_length // 4
            content_str = f"{content_str[:quarter]}...[truncated {len(content_str) - max_length} chars]...{content_str[-quarter:]}"

        # Clean up whitespace and limit lines
        lines = content_str.split("\n")
        compressed_lines = []

        for line in lines[:20]:  # Limit to first 20 lines
            # Remove leading/trailing whitespace and normalize internal spaces
            cleaned = " ".join(line.split())
            if cleaned:  # Skip empty lines
                compressed_lines.append(cleaned)

        if len(lines) > 20:
            compressed_lines.append(f"[{len(lines) - 20} more lines omitted]")

        # Join with single newlines
        compressed = "\n".join(compressed_lines)

        # Try to parse as JSON if it looks like JSON
        if compressed.startswith("{") or compressed.startswith("["):
            try:
                parsed = json.loads(compressed)
                # If it's JSON, apply structure compression with smaller limit
                return self._compress_attribute_value(
                    "content", parsed, max_length=500
                )
            except json.JSONDecodeError:
                # Not valid JSON, keep as string
                pass

        return compressed if compressed else content_str

    def _compress_tool_calls(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Compress tool calls while preserving essential info."""
        compressed = []
        for call in tool_calls:  # Process all tool calls
            if isinstance(call, dict):
                compressed.append({
                    "name": call.get(
                        "name", call.get("function", {}).get("name", "unknown")
                    ),
                    "args_summary": self._summarize_content(
                        call.get(
                            "arguments",
                            call.get("function", {}).get("arguments", {}),
                        )
                    ),
                })
        return compressed

    def _extract_span_interactions(
        self, spans: List[Any]
    ) -> List[Dict[str, Any]]:
        """Extract interactions from spans."""
        interactions = []

        for span in spans:
            if not isinstance(span, dict):
                continue

            # Look for LLM/agent interactions
            if (
                "llm" in span.get("span_name", "").lower()
                or "agent" in span.get("span_name", "").lower()
            ):
                attrs = span.get("span_attributes", {})

                interaction = {
                    "type": "span_interaction",
                    "span_name": span.get("span_name"),
                    "span_id": span.get("span_id"),
                }

                # Extract input/output messages
                for key in attrs:
                    if "input" in key.lower() and "message" in key.lower():
                        interaction["input"] = self._compress_message_content(
                            attrs[key]
                        )
                        break

                for key in attrs:
                    if "output" in key.lower() and "message" in key.lower():
                        interaction["output"] = self._compress_message_content(
                            attrs[key]
                        )
                        break

                if "input" in interaction or "output" in interaction:
                    interactions.append(interaction)

            # Recursively check child spans
            if "child_spans" in span:
                child_interactions = self._extract_span_interactions(
                    span["child_spans"]
                )
                interactions.extend(child_interactions)

        return interactions

    def _extract_key_decisions_with_context(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key decisions with their context."""
        decisions = []

        # Look for decision points with context
        for key in ["decisions", "choices", "selected", "actions"]:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    for item in value:
                        decision = {
                            "type": key,
                            "value": self._summarize_content(item),
                        }

                        # Try to add context if available
                        if isinstance(item, dict):
                            if "reason" in item:
                                decision["reason"] = self._summarize_content(
                                    item["reason"]
                                )
                            if "context" in item:
                                decision["context"] = self._summarize_content(
                                    item["context"]
                                )

                        decisions.append(decision)
                elif value:
                    decisions.append({
                        "type": key,
                        "value": self._summarize_content(value),
                    })

        # Extract tool/function decisions from spans
        if "spans" in data:
            tool_decisions = self._extract_tool_decisions(data["spans"])
            decisions.extend(tool_decisions)

            # Also look for tool usage directly in span attributes
            for span in data.get("spans", []):
                if isinstance(span, dict):
                    # Check for tool at top level (from our conversion)
                    if "tool" in span and span["tool"]:
                        decisions.append({
                            "type": "tool_usage",
                            "span_name": span.get("span_name", "unknown"),
                            "tool": span["tool"],
                        })

                    # Check for actions or decisions in attributes
                    if "span_attributes" in span:
                        attrs = span["span_attributes"]
                        if isinstance(attrs, dict):
                            for key in [
                                "action",
                                "decision",
                                "query",
                                "operation",
                            ]:
                                if key in attrs and attrs[key]:
                                    decisions.append({
                                        "type": key,
                                        "span_name": span.get(
                                            "span_name", "unknown"
                                        ),
                                        "value": attrs[key],
                                    })

        # Deduplicate decisions
        seen = set()
        unique_decisions = []
        for decision in decisions:
            # Create a hashable key for the decision
            key = str(decision)
            if key not in seen:
                seen.add(key)
                unique_decisions.append(decision)

        return unique_decisions  # Return unique decisions

    def _extract_tool_decisions(self, spans: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool/function call decisions from spans."""
        decisions = []

        for span in spans:
            if not isinstance(span, dict):
                continue

            attrs = span.get("span_attributes", {})

            # Look for tool/function calls
            if "tool" in attrs or "function" in attrs:
                decision = {
                    "type": "tool_call",
                    "span_name": span.get("span_name"),
                    "tool": attrs.get("tool", attrs.get("function")),
                }

                # Add arguments if available
                for key in ["arguments", "args", "params"]:
                    if key in attrs:
                        decision["args"] = self._summarize_content(attrs[key])
                        break

                decisions.append(decision)

            # Recursively check child spans
            if "child_spans" in span:
                child_decisions = self._extract_tool_decisions(
                    span["child_spans"]
                )
                decisions.extend(child_decisions)

        return decisions

    def _has_issues(self, data: Dict[str, Any]) -> bool:
        """Check if there are any issues to extract."""
        issue_keys = [
            "errors",
            "warnings",
            "exceptions",
            "issues",
            "problems",
            "failures",
        ]
        for key in issue_keys:
            if key in data and data[key]:
                return True

        # Check spans for errors
        if "spans" in data:
            for span in data.get("spans", []):
                if isinstance(span, dict):
                    if "error" in span or "exception" in span:
                        return True

        return False

    def _extract_execution_flow(self, data: Dict[str, Any]) -> List[str]:
        """Legacy method - kept for compatibility."""
        return [
            item["name"] for item in self._extract_execution_flow_detailed(data)
        ]

    def _is_important_span(self, span_name: str) -> bool:
        """Determine if a span is important enough to keep."""
        important_keywords = [
            "agent",
            "plan",
            "execute",
            "decide",
            "search",
            "query",
            "analyze",
            "evaluate",
            "retrieve",
            "generate",
            "tool",
            "function",
            "api",
            "llm",
            "model",
        ]
        span_lower = span_name.lower()
        return any(keyword in span_lower for keyword in important_keywords)

    def _extract_agent_interactions(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract and compress agent interactions."""
        interactions = []

        # Extract messages if present
        if "messages" in data:
            messages = data["messages"]
            if isinstance(messages, list):
                # Group messages by agent/role and compress
                compressed_messages = self._compress_messages(messages)
                interactions.extend(compressed_messages)

        # Extract from spans if present
        if "spans" in data:
            for span in data.get("spans", []):
                if (
                    isinstance(span, dict)
                    and "llm" in span.get("span_name", "").lower()
                ):
                    interaction = self._extract_llm_interaction(span)
                    if interaction:
                        interactions.append(interaction)

        return interactions  # Return all interactions

    def _compress_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Compress a list of messages by removing redundancy."""
        compressed = []
        seen_messages = set()

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Create a hashable key for deduplication
                msg_key = (role, str(content))

                # Skip duplicate messages
                if msg_key in seen_messages:
                    continue
                seen_messages.add(msg_key)

                compressed.append({
                    "role": role,
                    "content": self._compress_message_content(content),
                })

        # Add note if duplicates were removed
        if len(compressed) < len(messages):
            compressed.append({
                "note": f"[{len(messages) - len(compressed)} duplicate messages removed]"
            })

        return compressed

    def _summarize_content(self, content: Any) -> Any:
        """Intelligently summarize content without truncation."""
        if not content:
            return None

        # For dicts, show structure with key info
        if isinstance(content, dict):
            # Keep the structure but compress values
            summary = {}
            for key, value in content.items():
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    # For complex nested structures, show type and size
                    if isinstance(value, dict):
                        summary[key] = f"<dict with {len(value)} keys>"
                    else:
                        summary[key] = f"<list with {len(value)} items>"
                else:
                    # Keep simple values
                    summary[key] = value
            return summary

        # For lists, deduplicate and show count
        if isinstance(content, list):
            if len(content) == 0:
                return []

            # Check if all items are similar
            if all(isinstance(item, dict) for item in content):
                # For list of dicts, keep unique structures
                unique_structures = []
                seen_keys = set()
                for item in content:
                    keys = tuple(sorted(item.keys()))
                    if keys not in seen_keys:
                        seen_keys.add(keys)
                        unique_structures.append(self._summarize_content(item))

                if len(unique_structures) < len(content):
                    unique_structures.append(
                        f"[{len(content)} total items with {len(unique_structures)} unique structures]"
                    )
                return unique_structures
            else:
                # For other lists, remove duplicates
                unique_items = []
                seen = set()
                for item in content:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        unique_items.append(item)

                if len(unique_items) < len(content):
                    unique_items.append(
                        f"[{len(content) - len(unique_items)} duplicate items removed]"
                    )
                return unique_items

        # For strings, normalize whitespace
        if isinstance(content, str):
            # Normalize whitespace
            return " ".join(content.split())

        # For other types, return as-is
        return content

    def _extract_llm_interaction(
        self, span: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract LLM interaction from a span."""
        attrs = span.get("span_attributes", {})

        # Look for input/output messages
        input_msg = attrs.get("llm.input_messages.0.message.content", "")
        output_msg = attrs.get("llm.output_messages.0.message.content", "")

        if input_msg or output_msg:
            return {
                "type": "llm_call",
                "input": self._summarize_content(input_msg),
                "output": self._summarize_content(output_msg),
            }

        return None

    def _extract_key_decisions(self, data: Dict[str, Any]) -> List[str]:
        """Extract key decisions made during execution."""
        decisions = []

        # Look for decision points in the data
        for key in ["decisions", "choices", "selected", "picked"]:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    decisions.extend([str(v) for v in value])
                else:
                    decisions.append(str(value))

        # Look for tool calls
        if "tool_calls" in data:
            for call in data.get("tool_calls", []):
                if isinstance(call, dict):
                    tool_name = call.get(
                        "name", call.get("function", "unknown")
                    )
                    decisions.append(f"tool_call: {tool_name}")

        return decisions

    def _compress_plan(self, plan: Any) -> Any:
        """
        Preserve plan completely - no compression for metrics evaluation.

        The plan is critical for metrics and should never be compressed or summarized.
        If context window is an issue, compress other data instead.
        """
        if not plan:
            return None

        # Log that we're preserving the complete plan
        logger.info(
            "Preserving complete plan for metrics evaluation (no compression)"
        )

        # Return plan exactly as-is - no compression applied
        return plan

    def _extract_issues(self, data: Dict[str, Any]) -> List[str]:
        """Extract errors, warnings, and issues."""
        issues = []

        # Look for error-related keys
        for key in ["errors", "warnings", "exceptions", "issues", "problems"]:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    issues.extend([str(v) for v in value])
                elif value:
                    issues.append(str(value))

        # Look for error spans
        if "spans" in data:
            for span in data.get("spans", []):
                if isinstance(span, dict):
                    if "error" in span or "exception" in span:
                        error_msg = span.get(
                            "error", span.get("exception", "unknown error")
                        )
                        issues.append(str(error_msg))

        return issues

    def _extract_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final results and outcomes."""
        results = {}

        # Look for result-related keys
        for key in ["output", "result", "response", "answer", "outcome"]:
            if key in data:
                results[key] = self._summarize_content(data[key])

        # Look for success/failure indicators
        for key in ["success", "completed", "failed", "status"]:
            if key in data:
                results[key] = data[key]

        return results


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
    compressor = TraceCompressor()

    if preserve_plan:
        return compressor.compress_trace_with_plan_priority(
            trace_data, target_token_limit
        )
    else:
        return compressor.compress_trace(trace_data)


def compress_multiple_traces(traces: List[Any]) -> List[Dict[str, Any]]:
    """
    Compress multiple traces efficiently.

    Args:
        traces: List of trace data to compress

    Returns:
        List of compressed traces
    """
    compressor = TraceCompressor()
    compressed_traces = []

    for trace in traces:
        compressed = compressor.compress_trace(trace)
        compressed_traces.append(compressed)

    return compressed_traces
