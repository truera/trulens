"""
Provider interface for trace-specific parsing and plan extraction.

This allows different app integrations (LangGraph, LangChain, etc.) to provide
their own trace parsing logic while keeping the core compression generic.
"""

from abc import ABC
from abc import abstractmethod
import json
import logging
import re
from typing import Any, Dict, List, Optional

from trulens.core.utils.trace_compression import DEFAULT_TOKEN_LIMIT
from trulens.core.utils.trace_compression import safe_truncate

logger = logging.getLogger(__name__)


class TraceProvider(ABC):
    """Abstract base class for trace provider-specific parsing."""

    @abstractmethod
    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """
        Check if this provider can handle the given trace data.

        Args:
            trace_data: Raw trace data to check

        Returns:
            True if this provider can parse this trace format
        """
        pass

    @abstractmethod
    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """
        Extract plan information from trace data.

        Args:
            trace_data: Raw trace data

        Returns:
            Plan data if found, None otherwise
        """
        pass

    @abstractmethod
    def extract_execution_flow(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract execution flow from trace data.

        Args:
            trace_data: Raw trace data

        Returns:
            List of execution steps
        """
        pass

    @abstractmethod
    def extract_agent_interactions(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract agent interactions from trace data.

        Args:
            trace_data: Raw trace data

        Returns:
            List of agent interactions
        """
        pass

    def compress_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress trace data while preserving essential information.

        Args:
            trace_data: Raw trace data to compress

        Returns:
            Compressed trace data
        """

        compressed = {}

        # Always preserve plan first - use extract_plan method
        plan = self.extract_plan(trace_data)
        if plan:
            compressed["plan"] = plan

        # Extract tool execution evidence for plan adherence verification
        tool_evidence = self._extract_tool_execution_evidence(trace_data)
        if tool_evidence:
            compressed["tool_execution_evidence"] = tool_evidence

        # Also preserve any top-level keys containing "plan" (case-insensitive)
        for key, value in trace_data.items():
            if (
                "plan" in key.lower() and key.lower() != "plan"
            ):  # Don't duplicate the main plan
                compressed[key] = value

        # Always extract execution flow - useful alongside plan for adherence evaluation
        flow = self.extract_execution_flow(trace_data)
        if flow:
            compressed["execution_flow"] = flow

        # Add agent interactions
        interactions = self.extract_agent_interactions(trace_data)
        if interactions:
            compressed["agent_interactions"] = interactions

        # Add basic trace info
        if "trace_id" in trace_data:
            compressed["trace_id"] = trace_data["trace_id"]
        if "metadata" in trace_data:
            compressed["metadata"] = trace_data["metadata"]
        return compressed

    def _extract_tool_execution_evidence(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tool execution evidence for plan adherence verification.

        Extracts from multiple formats:
        - TruLens OTEL semantic conventions (ai.observability.*)
        - Snowflake Cortex Agent tool_use/tool_result in AIMessage content
        - LangGraph Command structures

        This is a base implementation that can be overridden by specific providers.
        """
        if "spans" not in trace_data:
            return None

        tool_evidence: Dict[str, List[Any]] = {
            "tool_calls": [],
            "node_outputs": [],
            "execution_sequence": [],
        }

        # TruLens OTEL semantic convention prefixes
        MCP_PREFIX = "ai.observability.mcp."
        GRAPH_NODE_PREFIX = "ai.observability.graph_node."
        GRAPH_TASK_PREFIX = "ai.observability.graph_task."
        CALL_PREFIX = "ai.observability.call."

        # Key attributes to extract
        MCP_TOOL_NAME = MCP_PREFIX + "tool_name"
        MCP_INPUT_ARGS = MCP_PREFIX + "input_arguments"
        MCP_OUTPUT = MCP_PREFIX + "output_content"

        GRAPH_NODE_NAME = GRAPH_NODE_PREFIX + "node_name"
        GRAPH_OUTPUT_STATE = GRAPH_NODE_PREFIX + "output_state"
        GRAPH_LATEST_MSG = GRAPH_NODE_PREFIX + "latest_message"

        CALL_FUNCTION = CALL_PREFIX + "function"
        CALL_RETURN = CALL_PREFIX + "return"

        for span in trace_data.get("spans", []):
            if not isinstance(span, dict):
                continue

            span_name = span.get("span_name", "")
            attrs = span.get("span_attributes", {})

            if not isinstance(attrs, dict):
                continue

            # 1. Extract MCP tool calls (OTEL format)
            if MCP_TOOL_NAME in attrs or MCP_OUTPUT in attrs:
                tool_call = {"span": span_name}
                if MCP_TOOL_NAME in attrs:
                    tool_call["tool_name"] = str(attrs[MCP_TOOL_NAME])
                if MCP_INPUT_ARGS in attrs:
                    val = str(attrs[MCP_INPUT_ARGS])
                    tool_call["input"] = self._safe_truncate(val, 1000)
                if MCP_OUTPUT in attrs:
                    val = str(attrs[MCP_OUTPUT])
                    tool_call["output"] = self._safe_truncate(val, 2000)
                tool_evidence["tool_calls"].append(tool_call)

            # 2. Extract graph node outputs (OTEL format)
            # AND parse for embedded tool calls in any format
            if GRAPH_OUTPUT_STATE in attrs or GRAPH_NODE_NAME in attrs:
                node_output = {"span": span_name}
                if GRAPH_NODE_NAME in attrs:
                    node_output["node_name"] = str(attrs[GRAPH_NODE_NAME])
                if GRAPH_OUTPUT_STATE in attrs:
                    val = str(attrs[GRAPH_OUTPUT_STATE])
                    node_output["output_state"] = self._safe_truncate(val, 3000)

                    # Parse output_state for embedded tool calls (any format)
                    if self._has_tool_call_indicators(val):
                        tool_calls_found = self._extract_embedded_tool_calls(
                            val, span_name
                        )
                        tool_evidence["tool_calls"].extend(tool_calls_found)
                    if self._has_tool_result_indicators(val):
                        tool_results_found = (
                            self._extract_embedded_tool_results(val, span_name)
                        )
                        tool_evidence["execution_sequence"].extend(
                            tool_results_found
                        )

                if GRAPH_LATEST_MSG in attrs:
                    val = str(attrs[GRAPH_LATEST_MSG])
                    node_output["latest_message"] = self._safe_truncate(
                        val, 2000
                    )

                    # Also check latest_message for embedded tool patterns
                    if self._has_tool_call_indicators(val):
                        tool_calls_found = self._extract_embedded_tool_calls(
                            val, span_name
                        )
                        tool_evidence["tool_calls"].extend(tool_calls_found)
                    if self._has_tool_result_indicators(val):
                        tool_results_found = (
                            self._extract_embedded_tool_results(val, span_name)
                        )
                        tool_evidence["execution_sequence"].extend(
                            tool_results_found
                        )

                tool_evidence["node_outputs"].append(node_output)

            # 3. Extract graph task outputs
            task_output_state = GRAPH_TASK_PREFIX + "output_state"
            task_name = GRAPH_TASK_PREFIX + "task_name"
            if task_output_state in attrs or task_name in attrs:
                task_output = {"span": span_name, "type": "graph_task"}
                if task_name in attrs:
                    task_output["task_name"] = str(attrs[task_name])
                if task_output_state in attrs:
                    val = str(attrs[task_output_state])
                    task_output["output_state"] = self._safe_truncate(val, 2000)
                tool_evidence["execution_sequence"].append(task_output)

            # 4. Extract function call returns
            if CALL_RETURN in attrs or CALL_FUNCTION in attrs:
                call_info = {"span": span_name, "type": "function_call"}
                if CALL_FUNCTION in attrs:
                    call_info["function"] = str(attrs[CALL_FUNCTION])
                if CALL_RETURN in attrs:
                    val = str(attrs[CALL_RETURN])
                    call_info["return_value"] = self._safe_truncate(val, 2000)
                tool_evidence["execution_sequence"].append(call_info)

            # 5. Scan all other attributes for embedded tool calls
            for key, value in attrs.items():
                if not isinstance(value, str):
                    continue

                # Look for embedded tool patterns in any attribute value
                if self._has_tool_call_indicators(value):
                    tool_calls_found = self._extract_embedded_tool_calls(
                        value, span_name
                    )
                    tool_evidence["tool_calls"].extend(tool_calls_found)

                if self._has_tool_result_indicators(value):
                    tool_results_found = self._extract_embedded_tool_results(
                        value, span_name
                    )
                    tool_evidence["execution_sequence"].extend(
                        tool_results_found
                    )

                # Also check for Command structures (LangGraph routing)
                if len(value) >= 100:
                    key_lower = key.lower()

                    if "Command(" in value and (
                        "goto" in value or "update" in value
                    ):
                        # Extract tool evidence from Command content
                        if self._has_tool_call_indicators(value):
                            tool_calls_found = (
                                self._extract_embedded_tool_calls(
                                    value, span_name
                                )
                            )
                            tool_evidence["tool_calls"].extend(tool_calls_found)
                        if self._has_tool_result_indicators(value):
                            tool_results_found = (
                                self._extract_embedded_tool_results(
                                    value, span_name
                                )
                            )
                            tool_evidence["execution_sequence"].extend(
                                tool_results_found
                            )

                        tool_evidence["execution_sequence"].append({
                            "span": span_name,
                            "type": "command",
                            "attribute": key,
                            "content": self._safe_truncate(value, 1500),
                        })
                    # Look for any output-like attributes
                    elif any(
                        pattern in key_lower
                        for pattern in [
                            "output",
                            "result",
                            "return",
                            "response",
                        ]
                    ):
                        tool_evidence["execution_sequence"].append({
                            "span": span_name,
                            "attribute": key,
                            "content": self._safe_truncate(value, 1500),
                        })

        # Only return if we found actual evidence
        has_evidence = (
            tool_evidence["tool_calls"]
            or tool_evidence["node_outputs"]
            or tool_evidence["execution_sequence"]
        )

        return tool_evidence if has_evidence else None

    def _has_tool_call_indicators(self, content: str) -> bool:
        """
        Check if content contains indicators of tool calls.

        Generic patterns that indicate tool invocations across frameworks:
        - 'name': combined with 'input': or 'arguments':
        - 'tool_use' or 'tool_call' type markers
        - 'function': with nested structure
        """
        # Must have a name field
        if "'name':" not in content and '"name":' not in content:
            return False

        # And one of these indicators
        indicators = [
            "'input':",  # Common input pattern
            '"input":',
            "'arguments':",  # OpenAI-style
            '"arguments":',
            "tool_use",  # Anthropic/generic
            "tool_call",  # OpenAI-style
            "'function':",  # Function call pattern
        ]
        return any(ind in content for ind in indicators)

    def _has_tool_result_indicators(self, content: str) -> bool:
        """
        Check if content contains indicators of tool results.

        Generic patterns that indicate tool responses:
        - 'status': 'success' or 'error'
        - 'tool_result' or 'tool_response' markers
        - Result data patterns (doc_id, search_results, etc.)
        """
        indicators = [
            "'status':",  # Status field present
            '"status":',
            "tool_result",  # Anthropic-style
            "tool_response",  # Generic
            "search_results",  # Search tool results
            "'doc_id':",  # Document IDs
            "'record_id':",  # Record IDs
        ]
        return any(ind in content for ind in indicators)

    def _safe_truncate(self, s: str, max_len: int) -> str:
        """
        Safely truncate a string without breaking JSON structure.
        Delegates to the shared safe_truncate utility.
        """
        return safe_truncate(s, max_len)

    def _extract_embedded_tool_calls(
        self, content: str, span_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tool calls embedded in message content.

        Detects common patterns for tool invocations:
        - {'type': 'tool_use', 'name': '...', 'input': {...}}
        - {'type': 'tool_call', 'function': {'name': '...', 'arguments': ...}}
        - {'tool_use': {'name': '...', ...}}
        """

        tool_calls = []

        # Generic patterns for tool names in various formats
        name_patterns = [
            r"'name':\s*'([^']+)'",  # 'name': 'tool_name'
            r'"name":\s*"([^"]+)"',  # "name": "tool_name"
            r"'function':\s*\{[^}]*'name':\s*'([^']+)'",  # function.name
        ]

        # Metadata keys to exclude (not actual tool names)
        excluded = {
            "thinking",
            "text",
            "json",
            "citation",
            "annotation",
            "type",
            "content",
        }

        seen_tools = set()
        for pattern in name_patterns:
            for tool_name in re.findall(pattern, content):
                tool_lower = tool_name.lower()
                if (
                    tool_name not in seen_tools
                    and tool_lower not in excluded
                    and not tool_lower.endswith("_citation")
                ):
                    seen_tools.add(tool_name)

                    tool_call = {"span": span_name, "tool_name": tool_name}

                    # Try to extract input/arguments - just get the query value, not full JSON
                    query_match = re.search(
                        rf"'name':\s*'{re.escape(tool_name)}'[^}}]*'query':\s*'([^']*)'",
                        content,
                    )
                    if query_match:
                        tool_call["query"] = self._safe_truncate(
                            query_match.group(1), 200
                        )

                    tool_calls.append(tool_call)

        return tool_calls

    def _extract_embedded_tool_results(
        self, content: str, span_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tool results embedded in message content.

        Captures:
        - Tool execution status (success/error)
        - Result data summaries (row counts, sample values)
        - Error messages when present
        """
        import re

        tool_results = []

        # Find tool result blocks - match tool_result with name and status
        # This handles nested structures like {'tool_result': {..., 'name': 'X', 'status': 'success'}}
        result_pattern = r"'tool_result':\s*\{[^}]*'name':\s*'([^']+)'[^}]*'status':\s*'([^']+)'"
        for match in re.finditer(result_pattern, content):
            tool_name = match.group(1)
            status = match.group(2).lower()

            # Skip metadata types
            if tool_name.lower() in {"thinking", "text", "json", "annotation"}:
                continue
            if tool_name.lower().endswith("_citation"):
                continue

            tool_results.append({
                "span": span_name,
                "type": "tool_result",
                "tool_name": tool_name,
                "status": status,
            })

        # Fallback: find standalone status patterns if no structured results found
        if not tool_results:
            status_pattern = r"'status':\s*'(success|error|failed|completed)'"
            name_pattern = r"'name':\s*'([^']+)'"

            statuses = re.findall(status_pattern, content, re.IGNORECASE)
            excluded = {
                "thinking",
                "text",
                "json",
                "annotation",
                "type",
                "content",
            }
            names = [
                n
                for n in re.findall(name_pattern, content)
                if n.lower() not in excluded
                and not n.lower().endswith("_citation")
            ]

            seen = set()
            for i, status in enumerate(statuses):
                tool_name = names[i] if i < len(names) else "tool"
                key = f"{tool_name}_{status}"
                if key not in seen:
                    seen.add(key)
                    tool_results.append({
                        "span": span_name,
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "status": status.lower(),
                    })

        # Extract result_set summaries (shows data was actually retrieved)
        # Pattern: 'result_set': {'data': [...], 'numRows': N, ...}
        num_rows_match = re.search(r"'numRows':\s*(\d+)", content)
        if num_rows_match:
            num_rows = int(num_rows_match.group(1))

            # Extract column names from rowType metadata
            col_names = re.findall(r"'name':\s*'([A-Z][A-Z_0-9]+)'", content)
            # Filter to likely SQL column names
            columns = list(dict.fromkeys(c for c in col_names if len(c) > 2))[
                :8
            ]

            result_summary = {
                "span": span_name,
                "type": "query_result",
                "rows_returned": num_rows,
            }
            if columns:
                result_summary["columns"] = columns
            tool_results.append(result_summary)

        # Extract sample data values from result sets
        # Look for common ID patterns in data arrays
        data_patterns = [
            (r"'(CUST_\d+)'", "customer_id"),
            (r"'(TKT_\d+)'", "ticket_id"),
            (r"'doc_id':\s*'([^']+)'", "doc_id"),
        ]

        for pattern, id_type in data_patterns:
            ids = re.findall(pattern, content)
            if ids:
                unique_ids = list(dict.fromkeys(ids))[:10]  # Preserve order
                tool_results.append({
                    "span": span_name,
                    "type": "data_retrieved",
                    "id_type": id_type,
                    "sample_values": unique_ids,
                    "count": len(set(ids)),
                })

        # Extract error details when present
        if "'status': 'error'" in content or '"status": "error"' in content:
            # Look for error message
            error_match = re.search(r"'Message':\s*'([^']{1,200})", content)
            if error_match:
                tool_results.append({
                    "span": span_name,
                    "type": "error_detail",
                    "message": error_match.group(1),
                })

        return tool_results

    def compress_with_plan_priority(
        self,
        trace_data: Dict[str, Any],
        target_token_limit: int = DEFAULT_TOKEN_LIMIT,
    ) -> Dict[str, Any]:
        """
        Compress trace with plan preservation as highest priority.
        If context window is exceeded, compress other data more aggressively.

        Args:
            trace_data: Raw trace data to compress
            target_token_limit: Target token limit for context window management

        Returns:
            Compressed trace data with plan preservation prioritized
        """

        # First, try normal compression but preserve plan
        compressed = self.compress_trace(trace_data)

        # Estimate tokens (rough approximation)
        estimated_tokens = len(json.dumps(compressed, default=str)) // 4
        if estimated_tokens <= target_token_limit:
            return compressed

        # If still too large, compress non-plan data more aggressively
        # Extract ALL keys containing "plan" first
        plan_keys = {}
        plan_tokens = 0

        for key, value in compressed.items():
            if "plan" in key.lower():
                plan_keys[key] = value
                key_tokens = len(json.dumps(value, default=str)) // 4
                plan_tokens += key_tokens
                logger.debug(f"Found plan key '{key}': {key_tokens} tokens")

        # Rebuild with more aggressive compression for non-plan data
        result = {}

        # Always preserve ALL plan-related keys
        for key, value in plan_keys.items():
            result[key] = value

        # Add other data within budget
        used_tokens = plan_tokens

        for key, value in compressed.items():
            if "plan" in key.lower():
                continue  # Already added above

            value_tokens = len(json.dumps(value, default=str)) // 4
            logger.debug(f"Considering {key}: {value_tokens} tokens")

            if used_tokens + value_tokens <= target_token_limit:
                result[key] = value
                used_tokens += value_tokens
                logger.debug(f"Added {key}, now using {used_tokens} tokens")
            else:
                logger.debug(f"{key} too large, trying truncation")
                # Try to fit a truncated version
                if isinstance(value, list) and len(value) > 1:
                    # For lists, try with fewer items
                    for i in range(len(value) - 1, 0, -1):
                        truncated = value[:i]
                        truncated_tokens = (
                            len(json.dumps(truncated, default=str)) // 4
                        )
                        if used_tokens + truncated_tokens <= target_token_limit:
                            result[key] = truncated
                            used_tokens += truncated_tokens
                            break
        final_tokens = len(json.dumps(result, default=str)) // 4
        logger.debug(
            f"Final compressed trace: {final_tokens} tokens (plan: {plan_tokens})"
        )

        return result

    def _clean_plan_content(self, plan_value: Any) -> Any:
        """
        Clean plan content by removing debug messages, error logs, and other noise.

        Args:
            plan_value: Raw plan data that may contain debug messages

        Returns:
            Cleaned plan data with debug messages removed
        """

        if not isinstance(plan_value, str):
            # If it's not a string, convert to string for cleaning, then back
            plan_str = str(plan_value)
        else:
            plan_str = plan_value

        # Patterns to remove - ONLY obvious debug/error messages, be very conservative
        patterns_to_remove = [
            r"^Agent error: [^\n]*\n?",  # Remove lines that START with "Agent error: "
            r"^DEBUG: [^\n]*\n?",  # Remove lines that START with "DEBUG: "
            r"^Query ID: [^\n]*\n?",  # Remove lines that START with "Query ID: "
        ]

        cleaned_plan = plan_str

        for pattern in patterns_to_remove:
            cleaned_plan = re.sub(
                pattern, "", cleaned_plan, flags=re.IGNORECASE | re.MULTILINE
            )

        # Only remove completely empty lines, preserve all other content
        lines = cleaned_plan.split("\n")
        cleaned_lines = []

        for line in lines:
            # Keep the line if it has any content, even just whitespace
            # Only remove completely empty lines
            if line.strip():  # Keep any line with content
                cleaned_lines.append(line)

        # Join back with single newlines, preserve original formatting
        final_cleaned = "\n".join(cleaned_lines)

        # If the original was not a string, try to preserve the original type
        if not isinstance(plan_value, str):
            try:
                # Try to parse back to original format if it was JSON-like
                if plan_str.strip().startswith(("{", "[")):
                    return (
                        json.loads(final_cleaned)
                        if final_cleaned.strip()
                        else plan_value
                    )
            except Exception:
                pass

        return final_cleaned if final_cleaned.strip() else plan_value


class GenericTraceProvider(TraceProvider):
    """Generic trace provider for standard trace formats."""

    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """Generic provider handles any trace data as fallback."""
        return True

    def _extract_plan_from_spans(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Extract plan from spans, looking for Command structures and state attributes.
        This is used as a fallback when top-level plan keys contain debug messages.
        """
        if "spans" not in trace_data or not isinstance(
            trace_data["spans"], list
        ):
            logger.debug("No spans found in trace_data")
            return None

        logger.debug(f"Checking {len(trace_data['spans'])} spans for plan data")

        for i, span in enumerate(trace_data["spans"]):
            if not isinstance(span, dict):
                logger.debug(f"Span {i} is not a dict: {type(span)}")
                continue

            span_name = span.get("span_name", f"span_{i}")
            span_attrs = span.get("span_attributes", {})

            logger.debug(
                f"Span {i} '{span_name}' has {len(span_attrs) if isinstance(span_attrs, dict) else 0} attributes"
            )

            if not isinstance(span_attrs, dict):
                logger.debug(
                    f"Span {i} attributes not a dict: {type(span_attrs)}"
                )
                continue

            # Log all attribute keys for debugging
            attr_keys = list(span_attrs.keys())
            logger.debug(
                f"Span {i} '{span_name}' attribute keys: {attr_keys[:10]}{'...' if len(attr_keys) > 10 else ''}"
            )

            # Check for LangGraph-style state attributes
            state_keys = [
                "ai.observability.graph_node.output_state",
                "ai.observability.graph_node.input_state",
            ]

            # Collect all potential plans and prioritize them
            potential_plans = []

            for state_key in state_keys:
                if state_key in span_attrs:
                    state_value = span_attrs[state_key]
                    logger.debug(
                        f"Found {state_key} in {span_name}, type: {type(state_value)}, size: {len(str(state_value))}"
                    )
                    logger.debug(
                        f"State value preview: {str(state_value)[:200]}..."
                    )

                    # Check if it contains Command structure
                    if (
                        isinstance(state_value, str)
                        and "Command(" in state_value
                    ):
                        logger.debug(
                            f"Found Command structure in {span_name}.{state_key}"
                        )

                        # Prioritize Commands with 'plan' over Commands with 'messages'
                        priority = 0
                        if "'plan':" in state_value or '"plan":' in state_value:
                            priority = 100  # High priority for real plans
                            logger.debug(
                                "Command contains 'plan' key - HIGH PRIORITY"
                            )

                        elif (
                            "'messages':" in state_value
                            and "Agent error:" in state_value
                        ):
                            priority = 10  # Low priority for debug messages
                            logger.debug(
                                "Command contains debug messages - LOW PRIORITY"
                            )
                        else:
                            priority = 50  # Medium priority for other Commands
                            logger.debug(
                                "Command structure found - MEDIUM PRIORITY"
                            )

                        potential_plans.append({
                            "content": state_value,
                            "priority": priority,
                            "span_name": span_name,
                            "state_key": state_key,
                        })

                    # Also check for direct plan content (non-Command)
                    elif (
                        isinstance(state_value, str)
                        and "plan" in state_value.lower()
                        and len(state_value) > 50
                    ):
                        if "Agent error:" not in state_value:
                            logger.debug(
                                f"Found direct plan content in {span_name}.{state_key}"
                            )
                            potential_plans.append({
                                "content": state_value,
                                "priority": 70,  # Medium-high priority for direct plans
                                "span_name": span_name,
                                "state_key": state_key,
                            })

            # Sort by priority (highest first) and return the best plan
            if potential_plans:
                potential_plans.sort(key=lambda x: x["priority"], reverse=True)
                best_plan = potential_plans[0]
                logger.debug(
                    f"Selected best plan from {best_plan['span_name']}.{best_plan['state_key']} with priority {best_plan['priority']}"
                )

                # Try to extract structured plan from Command if applicable
                if "Command(" in best_plan["content"]:
                    if best_plan["priority"] >= 90:  # High priority plans
                        extracted_plan = self._extract_plan_from_command_string(
                            best_plan["content"]
                        )
                        if extracted_plan:
                            logger.debug(
                                "Successfully extracted structured plan from Command"
                            )
                            return extracted_plan

                # Return the raw content if extraction failed or not applicable
                return best_plan["content"]

            # Also check other attributes that might contain plans
            for attr_key, attr_value in span_attrs.items():
                if (
                    "plan" in attr_key.lower()
                    and isinstance(attr_value, str)
                    and len(attr_value) > 50
                ):
                    logger.debug(
                        f"Found plan-related attribute '{attr_key}' in {span_name}"
                    )
                    if "Agent error:" not in attr_value:
                        return attr_value

        logger.debug("No plan found in any spans")
        return None

    def _extract_plan_from_command_string(
        self, command_str: str
    ) -> Optional[str]:
        """
        Extract plan or plan from Command string using brace counting.
        This is a simplified version of the LangGraph provider logic.
        """
        try:
            import ast

            # Try multiple plan key patterns
            plan_markers = [
                '"execution_plan":',
                '"plan":',
            ]
            start_pos = -1

            for marker in plan_markers:
                pos = command_str.find(marker)
                if pos != -1:
                    start_pos = pos
                    break

            if start_pos == -1:
                logger.debug(
                    "🔍 GENERIC_PLAN_DEBUG: No plan markers found in Command string"
                )
                return None

            dict_start = command_str.find("{", start_pos)
            if dict_start == -1:
                return None

            brace_count = 0
            dict_end = dict_start

            for i in range(dict_start, len(command_str)):
                if command_str[i] == "{":
                    brace_count += 1
                elif command_str[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        dict_end = i
                        break

            if brace_count == 0:
                plan_str = command_str[dict_start : dict_end + 1]

                try:
                    plan_dict = ast.literal_eval(plan_str)
                    if isinstance(plan_dict, dict):
                        return self._format_plan_simple(plan_dict)
                except (ValueError, SyntaxError):
                    return plan_str

            return None
        except Exception as e:
            logger.debug(f"Error extracting plan: {e}")
            return None

    def _format_plan_simple(self, plan_dict: Dict[str, Any]) -> str:
        """
        Simple formatting of plan dictionary.
        """
        formatted_parts = []

        if "plan_summary" in plan_dict:
            formatted_parts.append(f"Plan Summary: {plan_dict['plan_summary']}")

        if "steps" in plan_dict and isinstance(plan_dict["steps"], list):
            formatted_parts.append("Steps:")
            for i, step in enumerate(plan_dict["steps"], 1):
                if isinstance(step, dict):
                    step_text = f"{i}. "
                    if "agent" in step:
                        step_text += f"Agent: {step['agent']} - "
                    if "purpose" in step:
                        step_text += f"Purpose: {step['purpose']}"
                    formatted_parts.append(step_text)

        if "expected_final_output" in plan_dict:
            formatted_parts.append(
                f"Expected Output: {plan_dict['expected_final_output']}"
            )

        return "\n".join(formatted_parts)

    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """Extract plan using generic field names."""
        logger.debug("GenericTraceProvider.extract_plan called")
        logger.debug(f"Available trace_data keys: {list(trace_data.keys())}")

        # Log first few keys and their types for debugging
        if trace_data:
            sample_keys = list(trace_data.keys())[:5]
            for key in sample_keys:
                value = trace_data[key]
                logger.debug(
                    f"Key '{key}' -> type: {type(value)}, size: {len(str(value))}"
                )

        # Check common plan field names
        plan_fields = ["plan", "plan", "agent_plan", "workflow_plan"]
        logger.debug(f"Checking plan fields: {plan_fields}")

        for field in plan_fields:
            if field in trace_data:
                plan_value = trace_data[field]

                # Check if this looks like debug messages - if so, skip it and look in spans instead
                plan_str = str(plan_value)
                if (
                    "Agent error:" in plan_str
                    and len(plan_str.split("Agent error:")) > 3
                ):
                    logger.debug(
                        f"Top-level '{field}' contains debug messages, checking spans instead"
                    )
                    # Look for better plan in spans first
                    span_plan = self._extract_plan_from_spans(trace_data)
                    if span_plan:
                        return span_plan
                    # If no span plan found, continue with cleaning the debug messages

                # Clean the plan by removing debug/error messages
                cleaned_plan = self._clean_plan_content(plan_value)
                cleaned_size = len(str(cleaned_plan))

                # If plan is extremely large (>10KB), truncate it aggressively
                if cleaned_size > 10000:
                    plan_str = str(cleaned_plan)

                    # Keep only first 5KB and add truncation notice
                    truncated_plan = (
                        plan_str[:5000]
                        + "... [PLAN TRUNCATED - ORIGINAL SIZE: "
                        + str(cleaned_size)
                        + " CHARS]"
                    )
                    return truncated_plan

                return cleaned_plan
            else:
                logger.debug(f"Field '{field}' not found in trace_data")

        # Check if this might be a LangGraph trace with spans
        if "spans" in trace_data:
            spans = trace_data["spans"]

            if isinstance(spans, list) and spans:
                # Check first few spans for plan-related data
                for i, span in enumerate(spans[:3]):
                    if isinstance(span, dict):
                        # Check span attributes
                        if "span_attributes" in span:
                            attrs = span["span_attributes"]
                            if isinstance(attrs, dict):
                                attr_keys = list(attrs.keys())
                                # Look for LangGraph state or plan-related attributes
                                for attr_key in attr_keys:
                                    if (
                                        "plan" in attr_key.lower()
                                        or "state" in attr_key.lower()
                                    ):
                                        attr_value = attrs[attr_key]
                                        logger.debug(
                                            f"PLAN_DEBUG: Found potential plan in span {i} attr '{attr_key}': {type(attr_value)}, size: {len(str(attr_value))}"
                                        )

        logger.debug("PLAN_DEBUG: No plan found in any generic field or spans")
        return None

    def extract_execution_flow(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract execution flow using generic span structure."""
        flow = []

        if "spans" in trace_data:
            for i, span in enumerate(trace_data["spans"]):
                if isinstance(span, dict):
                    flow_item = {
                        "step": i + 1,
                        "name": span.get("span_name", "unknown"),
                        "type": span.get("span_type", "step"),
                    }
                    flow.append(flow_item)

        return flow

    def extract_agent_interactions(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract agent interactions using generic message structure."""
        interactions = []

        if "messages" in trace_data:
            messages = trace_data["messages"]
            if isinstance(messages, list):
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        interaction = {
                            "index": i,
                            "role": msg.get("role", "unknown"),
                            "content": str(msg.get("content", ""))[
                                :500
                            ],  # Limit content
                        }
                        interactions.append(interaction)

        return interactions


class TraceProviderRegistry:
    """Registry for trace providers with priority ordering."""

    def __init__(self):
        self._providers: List[TraceProvider] = []
        # Always register generic provider as fallback
        self.register_provider(GenericTraceProvider())

    def register_provider(self, provider: TraceProvider):
        """Register a trace provider. Last registered has highest priority."""
        self._providers.insert(0, provider)  # Insert at beginning for priority

    def get_provider(self, trace_data: Dict[str, Any]) -> TraceProvider:
        """Get the first provider that can handle the trace data."""

        for i, provider in enumerate(self._providers):
            can_handle = provider.can_handle(trace_data)

            if can_handle:
                return provider

        # This should never happen since GenericTraceProvider handles everything
        fallback = self._providers[-1]
        return fallback


# Global registry instance
_trace_provider_registry = TraceProviderRegistry()


def register_trace_provider(provider: TraceProvider):
    """Register a trace provider globally."""
    _trace_provider_registry.register_provider(provider)


def get_trace_provider(trace_data: Dict[str, Any]) -> TraceProvider:
    """Get appropriate trace provider for the given data."""
    return _trace_provider_registry.get_provider(trace_data)
