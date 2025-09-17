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

        # 4. Extract plan if present (keep structure)
        if "plan" in data:
            compressed["plan"] = self._compress_plan(data["plan"])
        elif "spans" in data:
            # Try to extract plan from span attributes
            for span in data["spans"]:
                if isinstance(span, dict) and "plan" in span:
                    compressed["plan"] = self._compress_plan(span["plan"])
                    break
                elif isinstance(span, dict) and "span_attributes" in span:
                    attrs = span["span_attributes"]
                    if isinstance(attrs, dict) and "plan" in attrs:
                        compressed["plan"] = self._compress_plan(attrs["plan"])
                        break

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
        """Compress a plan while keeping essential structure."""
        if isinstance(plan, list):
            # Keep only step summaries
            compressed_plan = []
            for step in plan:  # Process all steps
                if isinstance(step, dict):
                    compressed_step = {
                        "step": step.get("step", step.get("id", "unknown")),
                        "action": self._summarize_content(
                            step.get("action", step.get("description", ""))
                        ),
                    }
                    compressed_plan.append(compressed_step)
                else:
                    compressed_plan.append(self._summarize_content(step))
            return compressed_plan
        elif isinstance(plan, dict):
            # Keep only key fields
            return {
                "summary": self._summarize_content(
                    plan.get("summary", plan.get("description", ""))
                ),
                "steps": self._compress_plan(plan.get("steps", [])),
            }
        else:
            return self._summarize_content(plan)

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


def compress_trace_for_feedback(trace_data: Any) -> Dict[str, Any]:
    """
    Convenience function to compress trace data for feedback functions.

    Args:
        trace_data: The trace data to compress

    Returns:
        Compressed trace data
    """
    compressor = TraceCompressor()
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
