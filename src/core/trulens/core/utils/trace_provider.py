"""
Provider interface for trace-specific parsing and plan extraction.

This allows different app integrations (LangGraph, LangChain, etc.) to provide
their own trace parsing logic while keeping the core compression generic.
"""

from abc import ABC
from abc import abstractmethod
import json
import logging
from typing import Any, Dict, List, Optional

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

        # Also preserve any top-level keys containing "plan" (case-insensitive)
        plan_keys_found = 0
        for key, value in trace_data.items():
            if (
                "plan" in key.lower() and key.lower() != "plan"
            ):  # Don't duplicate the main plan
                compressed[key] = value
                plan_keys_found += 1
        if plan_keys_found == 0 and not plan:
            # Add execution flow
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

    def compress_with_plan_priority(
        self, trace_data: Dict[str, Any], target_token_limit: int = 100000
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
                logger.info(
                    f"DEBUG: Found plan key '{key}': {key_tokens} tokens"
                )

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
            logger.info(f"DEBUG: Considering {key}: {value_tokens} tokens")

            if used_tokens + value_tokens <= target_token_limit:
                result[key] = value
                used_tokens += value_tokens
                logger.info(
                    f"DEBUG: Added {key}, now using {used_tokens} tokens"
                )
            else:
                logger.info(f"DEBUG: {key} too large, trying truncation")
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


class GenericTraceProvider(TraceProvider):
    """Generic trace provider for standard trace formats."""

    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """Generic provider handles any trace data as fallback."""
        return True

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

        import re

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
                import json

                if plan_str.strip().startswith(("{", "[")):
                    return (
                        json.loads(final_cleaned)
                        if final_cleaned.strip()
                        else plan_value
                    )
            except Exception:
                pass

        return final_cleaned if final_cleaned.strip() else plan_value

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
            logger.warning(
                "🔍 GENERIC_PLAN_DEBUG: No spans found in trace_data"
            )
            return None

        logger.warning(
            f"🔍 GENERIC_PLAN_DEBUG: Checking {len(trace_data['spans'])} spans for plan data"
        )

        for i, span in enumerate(trace_data["spans"]):
            if not isinstance(span, dict):
                logger.warning(
                    f"🔍 GENERIC_PLAN_DEBUG: Span {i} is not a dict: {type(span)}"
                )
                continue

            span_name = span.get("span_name", f"span_{i}")
            span_attrs = span.get("span_attributes", {})

            logger.warning(
                f"🔍 GENERIC_PLAN_DEBUG: Span {i} '{span_name}' has {len(span_attrs) if isinstance(span_attrs, dict) else 0} attributes"
            )

            if not isinstance(span_attrs, dict):
                logger.warning(
                    f"🔍 GENERIC_PLAN_DEBUG: Span {i} attributes not a dict: {type(span_attrs)}"
                )
                continue

            # Log all attribute keys for debugging
            attr_keys = list(span_attrs.keys())
            logger.warning(
                f"🔍 GENERIC_PLAN_DEBUG: Span {i} '{span_name}' attribute keys: {attr_keys[:10]}{'...' if len(attr_keys) > 10 else ''}"
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
                    logger.warning(
                        f"🔍 GENERIC_PLAN_DEBUG: Found {state_key} in {span_name}, type: {type(state_value)}, size: {len(str(state_value))}"
                    )
                    logger.warning(
                        f"🔍 GENERIC_PLAN_DEBUG: State value preview: {str(state_value)[:200]}..."
                    )

                    # Check if it contains Command structure
                    if (
                        isinstance(state_value, str)
                        and "Command(" in state_value
                    ):
                        logger.warning(
                            f"🔍 GENERIC_PLAN_DEBUG: Found Command structure in {span_name}.{state_key}"
                        )

                        # Prioritize Commands with 'plan' over Commands with 'messages'
                        priority = 0
                        if "'plan':" in state_value or '"plan":' in state_value:
                            priority = 100  # High priority for real plans
                            logger.warning(
                                "🔍 GENERIC_PLAN_DEBUG: Command contains 'plan' key - HIGH PRIORITY"
                            )
                        elif (
                            "'plan':" in state_value or '"plan":' in state_value
                        ):
                            priority = 90  # High priority for execution plans
                            logger.warning(
                                "🔍 GENERIC_PLAN_DEBUG: Command contains 'plan' key - HIGH PRIORITY"
                            )
                        elif (
                            "'messages':" in state_value
                            and "Agent error:" in state_value
                        ):
                            priority = 10  # Low priority for debug messages
                            logger.warning(
                                "🔍 GENERIC_PLAN_DEBUG: Command contains debug messages - LOW PRIORITY"
                            )
                        else:
                            priority = 50  # Medium priority for other Commands
                            logger.warning(
                                "🔍 GENERIC_PLAN_DEBUG: Command structure found - MEDIUM PRIORITY"
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
                            logger.warning(
                                f"🔍 GENERIC_PLAN_DEBUG: Found direct plan content in {span_name}.{state_key}"
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
                logger.warning(
                    f"🔍 GENERIC_PLAN_DEBUG: Selected best plan from {best_plan['span_name']}.{best_plan['state_key']} with priority {best_plan['priority']}"
                )

                # Try to extract structured plan from Command if applicable
                if "Command(" in best_plan["content"]:
                    if best_plan["priority"] >= 90:  # High priority plans
                        extracted_plan = (
                            self._extract_execution_plan_from_command_string(
                                best_plan["content"]
                            )
                        )
                        if extracted_plan:
                            logger.warning(
                                "🔍 GENERIC_PLAN_DEBUG: Successfully extracted structured plan from Command"
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
                    logger.warning(
                        f"🔍 GENERIC_PLAN_DEBUG: Found plan-related attribute '{attr_key}' in {span_name}"
                    )
                    if "Agent error:" not in attr_value:
                        return attr_value

        logger.warning("🔍 GENERIC_PLAN_DEBUG: No plan found in any spans")
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
            logger.warning(f"🔍 GENERIC_PLAN_DEBUG: Error extracting plan: {e}")
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

    def _extract_plan_from_command_string(
        self, command_str: str
    ) -> Optional[str]:
        """
        Extract the plan content from a Command string representation.
        Simplified version for GenericTraceProvider.
        """
        try:
            import ast

            # Use brace counting to find the plan content
            start_marker = "'plan':"
            start_pos = command_str.find(start_marker)

            if start_pos == -1:
                return None

            # Find the start of the dictionary after the colon
            dict_start = command_str.find("{", start_pos)
            if dict_start == -1:
                return None

            # Count braces to find the matching closing brace
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

                # Try to parse it as a Python literal
                try:
                    plan_dict = ast.literal_eval(plan_str)
                    if isinstance(plan_dict, dict):
                        # Format the execution plan nicely
                        formatted_plan = self._format_plan_simple(plan_dict)
                        return formatted_plan
                except (ValueError, SyntaxError):
                    # Return the raw string if parsing fails
                    return plan_str

        except Exception as e:
            logger.warning(f"🔍 GENERIC_PLAN_DEBUG: Error extracting plan: {e}")

        return None

    def _format_plan_simple(self, plan_dict: dict) -> str:
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
        logger.info("DEBUG: GenericTraceProvider.extract_plan called")
        logger.warning(
            f"🔍 GENERIC_PLAN_DEBUG: Available trace_data keys: {list(trace_data.keys())}"
        )

        # Log first few keys and their types for debugging
        if trace_data:
            sample_keys = list(trace_data.keys())[:5]
            for key in sample_keys:
                value = trace_data[key]
                logger.warning(
                    f"PLAN_DEBUG: Key '{key}' -> type: {type(value)}, size: {len(str(value))}"
                )

        # Check common plan field names
        plan_fields = ["plan", "plan", "agent_plan", "workflow_plan"]
        logger.info(f"DEBUG: Checking plan fields: {plan_fields}")

        for field in plan_fields:
            if field in trace_data:
                plan_value = trace_data[field]
                plan_size = len(str(plan_value))
                logger.warning(
                    f"🔍 GENERIC_PLAN_DEBUG: Plan found in generic field '{field}', type: {type(plan_value)}, size: {plan_size}"
                )
                logger.warning(
                    f"🔍 GENERIC_PLAN_DEBUG: Plan content preview: {str(plan_value)[:300]}..."
                )

                # Check if this looks like debug messages - if so, skip it and look in spans instead
                plan_str = str(plan_value)
                if (
                    "Agent error:" in plan_str
                    and len(plan_str.split("Agent error:")) > 3
                ):
                    logger.warning(
                        f"🔍 GENERIC_PLAN_DEBUG: Top-level '{field}' contains debug messages, checking spans instead"
                    )
                    # Look for better plan in spans first
                    span_plan = self._extract_plan_from_spans(trace_data)
                    if span_plan:
                        logger.warning(
                            "🔍 GENERIC_PLAN_DEBUG: Found better plan in spans, using that instead"
                        )
                        return span_plan
                    # If no span plan found, continue with cleaning the debug messages

                # Clean the plan by removing debug/error messages
                cleaned_plan = self._clean_plan_content(plan_value)
                cleaned_size = len(str(cleaned_plan))

                if cleaned_size != plan_size:
                    logger.warning(
                        f"PLAN_DEBUG: Plan cleaned from {plan_size} to {cleaned_size} chars (removed debug messages)"
                    )

                # If plan is extremely large (>10KB), truncate it aggressively
                if cleaned_size > 10000:
                    logger.warning(
                        f"PLAN_DEBUG: Plan is very large ({cleaned_size} chars), applying aggressive truncation"
                    )
                    plan_str = str(cleaned_plan)

                    # Keep only first 5KB and add truncation notice
                    truncated_plan = (
                        plan_str[:5000]
                        + "... [PLAN TRUNCATED - ORIGINAL SIZE: "
                        + str(cleaned_size)
                        + " CHARS]"
                    )
                    logger.warning(
                        f"PLAN_DEBUG: Plan truncated from {cleaned_size} to {len(truncated_plan)} chars"
                    )
                    return truncated_plan

                return cleaned_plan
            else:
                logger.info(f"DEBUG: Field '{field}' not found in trace_data")

        # Check if this might be a LangGraph trace with spans
        if "spans" in trace_data:
            spans = trace_data["spans"]
            logger.warning(
                f"PLAN_DEBUG: Found {len(spans) if isinstance(spans, list) else 'non-list'} spans, checking for LangGraph plan data"
            )

            if isinstance(spans, list) and spans:
                # Check first few spans for plan-related data
                for i, span in enumerate(spans[:3]):
                    if isinstance(span, dict):
                        span_name = span.get("span_name", "unknown")
                        logger.warning(
                            f"PLAN_DEBUG: Span {i} '{span_name}' keys: {list(span.keys())}"
                        )

                        # Check span attributes
                        if "span_attributes" in span:
                            attrs = span["span_attributes"]
                            if isinstance(attrs, dict):
                                attr_keys = list(attrs.keys())
                                logger.warning(
                                    f"PLAN_DEBUG: Span {i} attributes: {attr_keys}"
                                )

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
