"""
LangGraph-specific trace provider for plan extraction and parsing.

This module contains all LangGraph-specific logic for parsing traces,
extracting plans from Command structures, and understanding LangGraph state.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from trulens.core.utils.trace_provider import TraceProvider

logger = logging.getLogger(__name__)


class LangGraphTraceProvider(TraceProvider):
    """LangGraph-specific trace provider that understands Command structures and graph state."""

    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """
        Check if this is a LangGraph trace by looking for LangGraph-specific indicators.
        Checks ALL spans for indicators, not just the first few.
        """

        # OTEL semantic convention prefixes used by TruLens for LangGraph
        LANGGRAPH_INDICATORS = [
            "ai.observability.graph_node",
            "ai.observability.graph_task",
            "ai.observability.mcp",  # MCP tools often used with LangGraph
            "graph_node",
            "graph_task",
            "langgraph",
        ]

        # Look for LangGraph-specific span names or attributes
        if "spans" in trace_data:
            spans = trace_data["spans"]
            if isinstance(spans, list):
                # Check ALL spans, not just first 3
                for span in spans:
                    if isinstance(span, dict):
                        span_name = span.get("span_name", "")

                        # Check span name
                        if any(
                            indicator in span_name.lower()
                            for indicator in ["langgraph", "graph", "pregel"]
                        ):
                            logger.info(
                                f"LANGGRAPH_DEBUG: Found LangGraph span name: {span_name}"
                            )
                            return True

                        # Check for LangGraph/OTEL observability attributes
                        attrs = span.get("span_attributes", {})
                        if isinstance(attrs, dict):
                            for key in attrs.keys():
                                if any(
                                    indicator in key.lower()
                                    for indicator in LANGGRAPH_INDICATORS
                                ):
                                    logger.info(
                                        f"LANGGRAPH_DEBUG: Found LangGraph attribute: {key}"
                                    )
                                    return True

                            # Also check for Command structures in attribute values
                            for value in attrs.values():
                                if (
                                    isinstance(value, str)
                                    and "Command(" in value
                                ):
                                    logger.info(
                                        "LANGGRAPH_DEBUG: Found Command structure in attributes"
                                    )
                                    return True

        logger.warning(
            "LANGGRAPH_DEBUG: No LangGraph indicators found in any span"
        )
        return False

    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """
        Extract plan from LangGraph trace, handling Command structures and graph state.
        """
        plan = self._extract_direct_plan_fields(trace_data)
        if plan:
            return plan

        # Strategy 2: Look for plans in LangGraph state attributes
        plan = self._extract_plan_from_graph_state(trace_data)
        if plan:
            return plan

        # Strategy 3: Look for Command structures in span outputs
        plan = self._extract_plan_from_command_structures(trace_data)
        if plan:
            return plan

        return None

    def _extract_plan_from_command_string(
        self, command_str: str
    ) -> Optional[str]:
        """
        Extract the plan content from a Command string representation.

        Example input: "Command(update={'plan': {'plan_summary': '...', 'steps': [...]}, ...})"
        """
        try:
            import ast
            import re

            # Look for the plan part specifically
            if "plan" not in command_str:
                return None

            # Try to extract any plan-related dictionary
            # Use a more robust approach to find plan content
            plan_key_pattern = r"['\"]([^'\"]*plan[^'\"]*)['\"]:"
            plan_keys = re.findall(plan_key_pattern, command_str, re.IGNORECASE)

            if not plan_keys:
                return None

            # Use the first plan key found
            plan_key = plan_keys[0]
            start_marker = f"'{plan_key}':"
            start_pos = command_str.find(start_marker)

            # Also try double quotes if single quotes not found
            if start_pos == -1:
                start_marker = f'"{plan_key}":'
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
                logger.info(
                    f"LangGraph extracted plan string for key '{plan_key}': {plan_str[:200]}..."
                )

                # Try to parse it as a Python literal
                try:
                    plan_dict = ast.literal_eval(plan_str)
                    if isinstance(plan_dict, dict):
                        # Format the plan nicely
                        formatted_plan = self._format_plan(plan_dict)
                        return formatted_plan
                except (ValueError, SyntaxError):
                    # Return the raw string if parsing fails
                    return plan_str

            return None

        except Exception as e:
            logger.warning(f"LangGraph error extracting plan from command: {e}")

        return None

    def _format_plan(self, plan_dict: dict) -> str:
        """
        Format an plan dictionary into a readable string.
        """
        formatted_parts = []

        # Add plan summary
        if "plan_summary" in plan_dict:
            formatted_parts.append(f"Plan Summary: {plan_dict['plan_summary']}")

        # Add steps
        if "steps" in plan_dict and isinstance(plan_dict["steps"], list):
            formatted_parts.append("Steps:")
            for i, step in enumerate(plan_dict["steps"], 1):
                if isinstance(step, dict):
                    step_text = f"{i}. "
                    if "agent" in step:
                        step_text += f"Agent: {step['agent']} - "
                    if "purpose" in step:
                        step_text += f"Purpose: {step['purpose']}"
                    if "expected_output" in step:
                        step_text += f" | Expected: {step['expected_output']}"
                    formatted_parts.append(step_text)

        # Add combination strategy
        if "combination_strategy" in plan_dict:
            formatted_parts.append(
                f"Strategy: {plan_dict['combination_strategy']}"
            )

        # Add expected final output
        if "expected_final_output" in plan_dict:
            formatted_parts.append(
                f"Expected Output: {plan_dict['expected_final_output']}"
            )

        return "\n".join(formatted_parts)

    def _extract_direct_plan_fields(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract plan from direct field names."""
        plan_fields = ["plan", "agent_plan", "workflow_plan"]

        for field in plan_fields:
            if field in trace_data:
                logger.info(f"LangGraph plan found in direct field '{field}'")
                raw_plan = trace_data[field]
                cleaned_plan = self._clean_plan_content(raw_plan)
                logger.info(
                    f"LangGraph plan cleaned: {len(str(raw_plan))} -> {len(str(cleaned_plan))} chars"
                )
                return cleaned_plan

        return None

    def _extract_plan_from_graph_state(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract plan from LangGraph graph node state attributes."""
        if "spans" not in trace_data:
            return None

        # Collect all potential plans with priority scoring
        potential_plans = []

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            span_name = span.get("span_name", "")
            attrs = span.get("span_attributes", {})
            if not isinstance(attrs, dict):
                continue

            # Check LangGraph-specific state attributes - prioritize output_state
            state_keys = [
                "ai.observability.graph_node.output_state",  # Check output_state first (more likely to have plan)
                "ai.observability.graph_node.input_state",
                "ai.observability.call.kwargs.input",
                "ai.observability.call.return",
            ]

            for state_key in state_keys:
                if state_key in attrs:
                    state_value = attrs[state_key]

                    # Calculate priority for this potential plan
                    priority = self._calculate_plan_priority(
                        state_value, state_key, span_name
                    )

                    if (
                        priority > 0
                    ):  # Only consider if it has some plan content
                        potential_plans.append({
                            "content": state_value,
                            "priority": priority,
                            "state_key": state_key,
                            "span_name": span_name,
                        })

        # Sort by priority (highest first) and extract the best plan
        if potential_plans:
            potential_plans.sort(key=lambda x: x["priority"], reverse=True)
            best_plan = potential_plans[0]

            # Extract the actual plan content
            plan = self._extract_plan_from_state_value(
                best_plan["content"],
                f"{best_plan['span_name']}.{best_plan['state_key']}",
            )
            if plan:
                return plan

        return None

    def _calculate_plan_priority(
        self, state_value: Any, state_key: str, span_name: str
    ) -> int:
        """Calculate priority score for a potential plan source."""
        priority = 0
        state_str = str(state_value)

        # Base priority by state key type
        if "output_state" in state_key:
            priority += 50  # High base priority for output state
        elif "input_state" in state_key:
            priority += 30  # Medium priority for input state
        elif "call.return" in state_key:
            priority += 10  # Low priority for call returns

        # High priority for Command structures with any key containing "plan"
        if "Command(" in state_str:
            # Look for any key containing "plan" as a substring
            import re

            plan_key_pattern = r"['\"]([^'\"]*plan[^'\"]*)['\"]:"
            plan_keys = re.findall(plan_key_pattern, state_str, re.IGNORECASE)

            if plan_keys:
                priority += (
                    100  # Very high priority for Command with plan-related key
                )
            else:
                priority += 40  # Medium priority for other Commands

        # Bonus for nodes that commonly contain planning logic (general patterns)
        planning_node_patterns = [
            "plan",
            "planning",
            "coordinator",
            "orchestrator",
            "manager",
            "router",
            "decision",
            "strategy",
            "workflow",
            "control",
        ]
        if any(
            pattern in span_name.lower() for pattern in planning_node_patterns
        ):
            priority += 15

        # Penalty for debug messages
        if "Agent error:" in state_str:
            debug_count = len(state_str.split("Agent error:"))
            penalty = min(debug_count * 10, 50)  # Cap penalty at 50
            priority -= penalty

        # Bonus for plan-related content (but less than Command structures)
        if any(
            keyword in state_str.lower()
            for keyword in ["plan_summary", "steps", "agent", "strategy"]
        ):
            priority += 20

        return priority

    def _extract_plan_from_state_value(
        self, state_value: Any, state_key: str
    ) -> Optional[Any]:
        """Extract plan from a state value (string or dict)."""
        if isinstance(state_value, str):
            # Check if this looks like a Command structure with any plan-related key
            if "Command(" in state_value:
                # Look for any key containing "plan" as a substring
                import re

                plan_key_pattern = r"['\"]([^'\"]*plan[^'\"]*)['\"]:"
                plan_keys = re.findall(
                    plan_key_pattern, state_value, re.IGNORECASE
                )

                if plan_keys:
                    extracted_plan = self._extract_plan_from_command_string(
                        state_value
                    )
                    if extracted_plan:
                        return extracted_plan

            # Try JSON parsing first
            try:
                parsed_state = json.loads(state_value)
                if isinstance(parsed_state, dict):
                    plan_fields = ["plan", "plan", "agent_plan"]
                    for field in plan_fields:
                        if field in parsed_state:
                            raw_plan = parsed_state[field]
                            cleaned_plan = self._clean_plan_content(raw_plan)
                            logger.info(
                                f"LangGraph plan cleaned from state: {len(str(raw_plan))} -> {len(str(cleaned_plan))} chars"
                            )
                            return cleaned_plan
            except json.JSONDecodeError as e:
                logger.debug(
                    f"🔍 LANGGRAPH_PLAN_DEBUG: JSON parsing failed: {e}"
                )

            # Check for plan-related content in string
            if "plan" in state_value.lower() and len(state_value) > 100:
                cleaned_plan = self._clean_plan_content(state_value)
                return cleaned_plan

        elif isinstance(state_value, dict):
            plan_fields = ["plan", "plan", "agent_plan"]
            for field in plan_fields:
                if field in state_value:
                    logger.info(f"LangGraph plan found in {state_key}.{field}")
                    return state_value[field]

        return None

    def _extract_plan_from_command_structures(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract plan from Command(...) structures in span outputs."""
        if "spans" not in trace_data:
            return None

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            # Check various span fields for Command structures
            for field in [
                "output",
                "result",
                "span_attributes",
                "processed_content",
            ]:
                if field in span:
                    value = span[field]
                    plan = self._extract_plan_from_command_value(
                        value, f"span.{field}"
                    )
                    if plan:
                        return plan

        return None

    def _extract_plan_from_command_value(
        self, value: Any, location: str
    ) -> Optional[Any]:
        """Extract plan from a value that might contain Command structures."""
        if isinstance(value, str):
            # Look for Command(...) patterns with plan
            if "Command(" in value and "plan" in value:
                logger.info(
                    f"LangGraph plan found in {location} Command structure"
                )
                # Try to extract just the plan part from the Command string
                extracted_plan = self._extract_plan_from_command_string(value)
                if extracted_plan:
                    cleaned_plan = self._clean_plan_content(extracted_plan)
                    logger.info(
                        f"LangGraph plan extracted and cleaned: {len(extracted_plan)} -> {len(str(cleaned_plan))} chars"
                    )
                    return cleaned_plan
                else:
                    # Fallback to cleaning the whole command
                    cleaned_plan = self._clean_plan_content(value)
                    logger.info(
                        f"LangGraph plan cleaned from Command: {len(value)} -> {len(str(cleaned_plan))} chars"
                    )
                    return cleaned_plan
            # Look for any plan-related content
            elif "plan" in value.lower() and len(value) > 100:
                logger.info(f"LangGraph plan found in {location} content")
                cleaned_plan = self._clean_plan_content(value)
                logger.info(
                    f"LangGraph plan cleaned from content: {len(value)} -> {len(str(cleaned_plan))} chars"
                )
                return cleaned_plan

        elif isinstance(value, dict):
            # Look for update structures: Command(update={...})
            if "update" in value:
                update_data = value["update"]
                if isinstance(update_data, dict):
                    plan_fields = ["plan", "plan", "agent_plan"]
                    for field in plan_fields:
                        if field in update_data:
                            logger.info(
                                f"LangGraph plan found in {location}.update.{field}"
                            )
                            raw_plan = update_data[field]
                            cleaned_plan = self._clean_plan_content(raw_plan)
                            logger.info(
                                f"LangGraph plan cleaned from update: {len(str(raw_plan))} -> {len(str(cleaned_plan))} chars"
                            )
                            return cleaned_plan

        return None

    def extract_execution_flow(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract detailed execution flow with reasoning chain from LangGraph trace."""
        logger.info(
            "DEBUG: LangGraphTraceProvider.extract_execution_flow called"
        )

        if "spans" not in trace_data:
            return []

        execution_flow = []
        for span in trace_data["spans"]:
            if isinstance(span, dict):
                span_name = span.get("span_name", "unknown")
                attrs = span.get("span_attributes", {})

                # Skip internal spans but be less aggressive
                if any(
                    skip in span_name.lower()
                    for skip in ["debug", "log", "trace", "monitor"]
                ):
                    continue

                # Extract detailed step information
                step = self._extract_detailed_step(span_name, attrs)
                if step:
                    execution_flow.append(step)

        logger.info(
            f"DEBUG: Extracted {len(execution_flow)} detailed execution steps"
        )
        return execution_flow

    def _extract_detailed_step(
        self, span_name: str, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract detailed step information including tool calls, reasoning, and data flow."""
        if not isinstance(attrs, dict):
            return None

        step = {
            "name": span_name,
            "type": self._classify_step_type(span_name),
        }

        # Extract tool calls and their results
        tool_info = self._extract_tool_information(attrs)
        if tool_info:
            step.update(tool_info)

        # Extract agent reasoning and decision points
        reasoning_info = self._extract_reasoning_information(attrs)
        if reasoning_info:
            step.update(reasoning_info)

        # Extract data transformations and calculations
        data_info = self._extract_data_transformations(attrs)
        if data_info:
            step.update(data_info)

        # Extract state transitions and agent handoffs
        transition_info = self._extract_state_transitions(attrs)
        if transition_info:
            step.update(transition_info)

        return (
            step if len(step) > 2 else None
        )  # Only return if we found meaningful content

    def _classify_step_type(self, span_name: str) -> str:
        """Classify the type of execution step."""
        name_lower = span_name.lower()

        if "supervisor" in name_lower or "coordinator" in name_lower:
            return "coordination"
        elif "agent" in name_lower:
            return "agent_execution"
        elif "tool" in name_lower or "search" in name_lower:
            return "tool_call"
        elif "langgraph" in name_lower:
            return "framework"
        elif "graph" == name_lower:
            return "graph_node"
        else:
            return "step"

    def _extract_tool_information(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract tool call details including inputs, outputs, and metadata."""
        tool_info = {}

        # Look for tool-related attributes
        for key, value in attrs.items():
            key_lower = key.lower()

            # Tool call detection
            if (
                "tool" in key_lower
                or "search" in key_lower
                or "call" in key_lower
            ):
                if isinstance(value, str) and len(value) > 10:
                    # Try to extract structured tool information
                    if "(" in value and ")" in value:
                        tool_info["tool_call"] = self._parse_tool_call(value)
                    else:
                        tool_info["tool_output"] = value[
                            :1000
                        ]  # Limit size but preserve detail

            # Function calls
            elif "function" in key_lower and isinstance(value, str):
                tool_info["function"] = value

            # Return values that might contain tool results
            elif "return" in key_lower and isinstance(value, str):
                if len(value) > 50:  # Substantial content
                    parsed_result = self._parse_tool_result(value)
                    if parsed_result:
                        tool_info["tool_result"] = parsed_result

        return tool_info if tool_info else None

    def _parse_tool_call(self, call_str: str) -> Dict[str, Any]:
        """Parse tool call string to extract structured information."""
        try:
            # Look for function name and parameters
            if "(" in call_str:
                func_name = call_str.split("(")[0].strip()

                # Try to extract parameters
                param_start = call_str.find("(")
                param_end = call_str.rfind(")")
                if param_start != -1 and param_end != -1:
                    params_str = call_str[param_start + 1 : param_end]

                    return {
                        "function": func_name,
                        "parameters": params_str[
                            :500
                        ],  # Limit but preserve detail
                        "raw_call": call_str[:200],
                    }

            return {"raw_call": call_str[:200]}
        except Exception:
            return {"raw_call": call_str[:200]}

    def _parse_tool_result(self, result_str: str) -> Optional[Dict[str, Any]]:
        """Parse tool result to extract key information."""
        try:
            # Try JSON parsing first
            if result_str.strip().startswith("{"):
                import json

                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict):
                        # Extract key fields for reasoning validation
                        return {
                            "type": "structured_result",
                            "data": {
                                k: v
                                for k, v in parsed.items()
                                if len(str(v)) < 500
                            },
                            "summary": str(parsed)[:300],
                        }
                except json.JSONDecodeError:
                    pass

            # Look for specific patterns that indicate data retrieval
            if any(
                keyword in result_str.lower()
                for keyword in [
                    "found",
                    "results",
                    "tickets",
                    "customers",
                    "data",
                ]
            ):
                return {
                    "type": "data_retrieval",
                    "content": self._safe_truncate(result_str, 1000),
                    "indicators": self._extract_data_indicators(result_str),
                }

            # For other substantial results
            if len(result_str) > 100:
                return {
                    "type": "execution_result",
                    "content": self._safe_truncate(result_str, 800),
                }

        except Exception:
            pass

        return None

    def _extract_data_indicators(self, text: str) -> Dict[str, Any]:
        """Extract key data indicators from text for logical consistency validation."""
        indicators = {}

        # Look for counts and quantities
        import re

        # Numbers that might be counts
        numbers = re.findall(
            r"\b(\d+)\s*(tickets?|customers?|results?|items?)\b",
            text,
            re.IGNORECASE,
        )
        if numbers:
            indicators["counts"] = {item: int(count) for count, item in numbers}

        # Ticket IDs
        ticket_ids = re.findall(r"TKT_\d+", text, re.IGNORECASE)
        if ticket_ids:
            indicators["ticket_ids"] = ticket_ids

        # Risk scores
        risk_scores = re.findall(r"(\d+(?:\.\d+)?)\s*[/\\]\s*(\d+)", text)
        if risk_scores:
            indicators["risk_scores"] = [
                f"{score}/{max_score}" for score, max_score in risk_scores
            ]

        # Keywords that indicate data types
        data_types = []
        for keyword in [
            "api",
            "churn",
            "satisfaction",
            "usage",
            "support",
            "complaints",
        ]:
            if keyword in text.lower():
                data_types.append(keyword)
        if data_types:
            indicators["data_types"] = data_types

        return indicators

    def _extract_reasoning_information(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract agent reasoning and decision points."""
        reasoning_info = {}

        for key, value in attrs.items():
            key_lower = key.lower()

            # Look for thinking or reasoning content
            if (
                "thinking" in key_lower
                or "reason" in key_lower
                or "analysis" in key_lower
            ):
                if isinstance(value, str) and len(value) > 20:
                    reasoning_info["thinking"] = value[:800]

            # Look for decision points
            elif "decision" in key_lower or "choice" in key_lower:
                if isinstance(value, str):
                    reasoning_info["decision"] = value[:400]

            # Look for state values that contain reasoning
            elif "state" in key_lower and isinstance(value, str):
                if any(
                    keyword in value.lower()
                    for keyword in [
                        "because",
                        "since",
                        "therefore",
                        "analysis",
                        "conclusion",
                    ]
                ):
                    reasoning_info["state_reasoning"] = value[:600]

        return reasoning_info if reasoning_info else None

    def _extract_data_transformations(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract data transformations and calculations."""
        data_info = {}

        for key, value in attrs.items():
            key_lower = key.lower()

            # Look for input/output patterns
            if "input" in key_lower and isinstance(value, str):
                if len(value) > 10:
                    data_info["input_data"] = value[:500]

            elif "output" in key_lower and isinstance(value, str):
                if len(value) > 10:
                    data_info["output_data"] = value[:500]

            # Look for calculations or scores
            elif any(
                calc_word in key_lower
                for calc_word in ["score", "calc", "result", "metric"]
            ):
                if isinstance(value, (str, int, float)):
                    data_info["calculation"] = str(value)[:200]

        return data_info if data_info else None

    def _extract_state_transitions(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract state transitions and agent handoffs."""
        transition_info = {}

        for key, value in attrs.items():
            key_lower = key.lower()

            # Look for agent transitions
            if "agent" in key_lower or "node" in key_lower:
                if isinstance(value, str) and len(value) < 100:
                    transition_info["agent_context"] = value

            # Look for Command structures that indicate transitions
            elif (
                "command" in key_lower
                or isinstance(value, str)
                and "Command(" in value
            ):
                if isinstance(value, str):
                    # Extract goto information
                    if "goto=" in value:
                        import re

                        goto_match = re.search(
                            r"goto=['\"]?([^'\")\s,]+)", value
                        )
                        if goto_match:
                            transition_info["next_agent"] = goto_match.group(1)

                    transition_info["command_context"] = value[:300]

        return transition_info if transition_info else None

    def extract_agent_interactions(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract detailed agent interactions with reasoning and data flow."""
        interactions = []

        if "spans" not in trace_data:
            return interactions

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            span_name = span.get("span_name", "unknown")
            attrs = span.get("span_attributes", {})
            if not isinstance(attrs, dict):
                continue

            # Extract detailed interaction information
            interaction = self._extract_detailed_interaction(span_name, attrs)
            if interaction:
                interactions.append(interaction)

        return interactions

    def _extract_detailed_interaction(
        self, span_name: str, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract detailed interaction including context, reasoning, and outcomes."""
        interaction = {
            "type": "agent_interaction",
            "node": span_name,
        }

        # Extract message content with more detail
        message_content = self._extract_message_content(attrs)
        if message_content:
            interaction.update(message_content)

        # Extract conversation context
        context_info = self._extract_conversation_context(attrs)
        if context_info:
            interaction.update(context_info)

        # Extract agent decisions and justifications
        decision_info = self._extract_agent_decisions(attrs)
        if decision_info:
            interaction.update(decision_info)

        return interaction if len(interaction) > 2 else None

    def _extract_message_content(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract message content with preservation of reasoning details."""
        content_info = {}

        for key, value in attrs.items():
            if not isinstance(value, str):
                continue

            key_lower = key.lower()

            # Primary message content
            if "message" in key_lower or "content" in key_lower:
                if len(value) > 20:
                    # Preserve more detail for reasoning validation
                    content_info["message"] = self._safe_truncate(value, 1500)

                    # Extract specific reasoning elements
                    reasoning_elements = self._extract_reasoning_elements(value)
                    if reasoning_elements:
                        content_info["reasoning_elements"] = reasoning_elements

            # Output state that might contain conclusions
            elif "output" in key_lower and len(value) > 50:
                # Look for conclusions and analysis
                if any(
                    keyword in value.lower()
                    for keyword in [
                        "summary",
                        "conclusion",
                        "analysis",
                        "assessment",
                    ]
                ):
                    content_info["analysis_output"] = self._safe_truncate(
                        value, 1000
                    )

        return content_info if content_info else None

    def _extract_reasoning_elements(self, text: str) -> Dict[str, Any]:
        """Extract specific reasoning elements for logical consistency validation."""
        elements = {}

        # Look for explicit claims and their support
        import re

        # Risk assessments
        risk_patterns = re.findall(
            r"(high|medium|low|critical)\s+(?:churn\s+)?risk",
            text,
            re.IGNORECASE,
        )
        if risk_patterns:
            elements["risk_assessments"] = risk_patterns

        # Numerical claims
        numerical_claims = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:[/\\]\s*\d+)?\s*(customers?|tickets?|percent|%)",
            text,
            re.IGNORECASE,
        )
        if numerical_claims:
            elements["numerical_claims"] = [
                f"{num} {unit}" for num, unit in numerical_claims
            ]

        # Causal relationships
        causal_indicators = []
        for pattern in [
            r"because\s+([^.]+)",
            r"due to\s+([^.]+)",
            r"caused by\s+([^.]+)",
            r"indicates?\s+([^.]+)",
        ]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            causal_indicators.extend(matches)
        if causal_indicators:
            elements["causal_claims"] = [
                claim.strip() for claim in causal_indicators
            ]

        # Evidence references
        evidence_refs = re.findall(
            r"(TKT_\d+|ticket\s+\w+|data shows?|results? indicate)",
            text,
            re.IGNORECASE,
        )
        if evidence_refs:
            elements["evidence_references"] = evidence_refs

        return elements

    def _extract_conversation_context(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract conversation context and flow."""
        context_info = {}

        for key, value in attrs.items():
            key_lower = key.lower()

            # Input state that shows what the agent received
            if "input" in key_lower and isinstance(value, str):
                if len(value) > 10:
                    context_info["input_context"] = value[:400]

            # Latest message that shows conversation flow
            elif "latest" in key_lower and isinstance(value, str):
                if len(value) > 10:
                    context_info["latest_message"] = value[:400]

        return context_info if context_info else None

    def _extract_agent_decisions(
        self, attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract agent decisions and their justifications."""
        decision_info = {}

        for key, value in attrs.items():
            if not isinstance(value, str):
                continue

            # Look for Command structures that show decisions
            if "Command(" in value:
                # Extract the decision and its target
                if "goto=" in value:
                    import re

                    goto_match = re.search(r"goto=['\"]?([^'\")\s,]+)", value)
                    if goto_match:
                        decision_info["agent_decision"] = {
                            "action": "agent_handoff",
                            "target": goto_match.group(1),
                            "context": value[:300],
                        }

                # Look for update information that shows what was decided
                if "update=" in value:
                    decision_info["state_update"] = value[:500]

        return decision_info if decision_info else None

    def _extract_tool_execution_evidence(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract explicit tool execution evidence for plan adherence verification.

        Uses TruLens OTEL semantic conventions to detect:
        - MCP tool calls (ai.observability.mcp.*)
        - Graph node outputs (ai.observability.graph_node.*)
        - Function call returns (ai.observability.call.*)
        """
        if "spans" not in trace_data:
            return None

        tool_evidence = {
            "tool_calls": [],
            "node_outputs": [],
            "execution_sequence": [],
        }

        # TruLens OTEL semantic convention prefixes
        MCP_PREFIX = "ai.observability.mcp."
        GRAPH_NODE_PREFIX = "ai.observability.graph_node."
        GRAPH_TASK_PREFIX = "ai.observability.graph_task."
        CALL_PREFIX = "ai.observability.call."

        # Key attributes to extract for each type
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

            # 1. Extract MCP tool calls (ai.observability.mcp.*)
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

            # 2. Extract graph node outputs (ai.observability.graph_node.*)
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

            # 3. Extract graph task outputs (ai.observability.graph_task.*)
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

            # 4. Extract function call returns (ai.observability.call.*)
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

                # Skip already processed OTEL keys
                if key.startswith((
                    MCP_PREFIX,
                    GRAPH_NODE_PREFIX,
                    GRAPH_TASK_PREFIX,
                    CALL_PREFIX,
                )):
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

                # Check for Command structures (LangGraph routing)
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
                    # Look for any output-like attributes not caught above
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
        import re

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

    def compress_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress trace with enhanced detail preservation for logical consistency."""
        logger.info("DEBUG: LangGraphTraceProvider.compress_trace called")

        compressed = {}

        # Extract plan with full detail
        plan = self.extract_plan(trace_data)
        if plan:
            compressed["plan"] = plan

        tool_evidence = self._extract_tool_execution_evidence(trace_data)
        if tool_evidence:
            compressed["tool_execution_evidence"] = tool_evidence

        # Extract enhanced execution flow with reasoning chain
        execution_flow = self.extract_execution_flow(trace_data)
        if execution_flow:
            compressed["execution_flow"] = execution_flow

        # Extract detailed agent interactions
        agent_interactions = self.extract_agent_interactions(trace_data)
        if agent_interactions:
            compressed["agent_interactions"] = agent_interactions

        # Add trace metadata
        if "trace_id" in trace_data:
            compressed["trace_id"] = trace_data["trace_id"]

        # Add evidence linking for logical consistency
        evidence_links = self._extract_evidence_links(trace_data)
        if evidence_links:
            compressed["evidence_links"] = evidence_links

        return compressed

    def compress_with_plan_priority(
        self, trace_data: Dict[str, Any], target_token_limit: int = 100000
    ) -> Dict[str, Any]:
        """Compress trace with plan priority and enhanced reasoning preservation."""

        # Use the enhanced compression by default since we have plenty of token budget
        return self.compress_trace(trace_data)

    def _extract_evidence_links(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract evidence links between claims and supporting data."""
        if "spans" not in trace_data:
            return None

        evidence_links = {}
        claims = []
        evidence = []

        # Collect claims and evidence from spans
        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            attrs = span.get("span_attributes", {})
            span_name = span.get("span_name", "")

            # Look for claims in outputs
            for key, value in attrs.items():
                if isinstance(value, str):
                    # Extract claims (statements that need evidence)
                    span_claims = self._extract_claims(value, span_name)
                    claims.extend(span_claims)

                    # Extract evidence (data that supports claims)
                    span_evidence = self._extract_evidence(value, span_name)
                    evidence.extend(span_evidence)

        if claims or evidence:
            evidence_links["claims"] = claims
            evidence_links["evidence"] = evidence
            evidence_links["links"] = self._link_claims_to_evidence(
                claims, evidence
            )

        return evidence_links if evidence_links else None

    def _extract_claims(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract claims that need evidence support."""
        claims = []
        import re

        # Risk assessment claims
        risk_claims = re.findall(
            r"(high|medium|low|critical)\s+(?:churn\s+)?risk(?:\s+\(([^)]+)\))?",
            text,
            re.IGNORECASE,
        )
        for risk_level, score in risk_claims:
            claims.append({
                "type": "risk_assessment",
                "claim": f"{risk_level} churn risk"
                + (f" ({score})" if score else ""),
                "source": source,
                "needs_evidence": ["data_source", "calculation_method"],
            })

        # Numerical claims
        numerical_claims = re.findall(
            r"(\d+)\s+(customers?|tickets?|complaints?)", text, re.IGNORECASE
        )
        for count, entity in numerical_claims:
            claims.append({
                "type": "numerical_claim",
                "claim": f"{count} {entity}",
                "source": source,
                "needs_evidence": ["data_source", "query_results"],
            })

        # Causal claims
        causal_patterns = [
            r"indicates?\s+([^.]+)",
            r"shows?\s+([^.]+)",
            r"suggests?\s+([^.]+)",
        ]
        for pattern in causal_patterns:
            causal_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in causal_matches:
                claims.append({
                    "type": "causal_claim",
                    "claim": match.strip(),
                    "source": source,
                    "needs_evidence": ["supporting_data", "logical_connection"],
                })

        return claims

    def _extract_evidence(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract evidence that can support claims."""
        evidence = []
        import re

        # Ticket IDs as evidence
        ticket_ids = re.findall(r"TKT_\d+", text, re.IGNORECASE)
        for ticket_id in ticket_ids:
            evidence.append({
                "type": "data_reference",
                "evidence": ticket_id,
                "source": source,
                "supports": ["numerical_claims", "data_retrieval"],
            })

        # Tool results as evidence
        if any(
            keyword in text.lower()
            for keyword in ["found", "retrieved", "search", "query"]
        ):
            evidence.append({
                "type": "tool_execution",
                "evidence": text[:300],
                "source": source,
                "supports": ["data_source", "query_results"],
            })

        # Calculations as evidence
        calc_patterns = re.findall(
            r"(\d+(?:\.\d+)?)\s*[+\-*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)",
            text,
        )
        for calc in calc_patterns:
            evidence.append({
                "type": "calculation",
                "evidence": f"{calc[0]} + {calc[1]} = {calc[2]}",
                "source": source,
                "supports": ["calculation_method", "numerical_claims"],
            })

        return evidence

    def _link_claims_to_evidence(
        self, claims: List[Dict], evidence: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Link claims to their supporting evidence."""
        links = []

        for claim in claims:
            supporting_evidence = []

            for evidence_item in evidence:
                # Check if evidence supports this type of claim
                if any(
                    need in evidence_item.get("supports", [])
                    for need in claim.get("needs_evidence", [])
                ):
                    supporting_evidence.append(evidence_item["evidence"])

            if supporting_evidence:
                links.append({
                    "claim": claim["claim"],
                    "claim_source": claim["source"],
                    "supporting_evidence": supporting_evidence,
                    "evidence_strength": "strong"
                    if len(supporting_evidence) >= 2
                    else "weak",
                })
            else:
                links.append({
                    "claim": claim["claim"],
                    "claim_source": claim["source"],
                    "supporting_evidence": [],
                    "evidence_strength": "none",
                })

        return links


def register_langgraph_provider():
    """Register the LangGraph trace provider."""
    from trulens.core.utils.trace_provider import register_trace_provider

    provider = LangGraphTraceProvider()
    register_trace_provider(provider)


# Auto-register when module is imported
register_langgraph_provider()
