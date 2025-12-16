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
        """
        logger.warning(
            "LANGGRAPH_DEBUG: LangGraphTraceProvider.can_handle called"
        )

        if not isinstance(trace_data, dict):
            logger.warning("LANGGRAPH_DEBUG: trace_data is not a dict")
            return False

        logger.warning(
            f"LANGGRAPH_DEBUG: trace_data keys: {list(trace_data.keys())}"
        )

        # Look for LangGraph-specific span names or attributes
        if "spans" in trace_data:
            spans = trace_data["spans"]
            logger.warning(
                f"LANGGRAPH_DEBUG: Found {len(spans) if isinstance(spans, list) else 'non-list'} spans"
            )

            if isinstance(spans, list):
                for i, span in enumerate(spans[:3]):  # Check first 3 spans
                    if isinstance(span, dict):
                        span_name = span.get("span_name", "")
                        logger.warning(
                            f"LANGGRAPH_DEBUG: Span {i} name: '{span_name}'"
                        )

                        if (
                            "langgraph" in span_name.lower()
                            or "graph" in span_name.lower()
                        ):
                            logger.warning(
                                f"LANGGRAPH_DEBUG: Found LangGraph span name: '{span_name}'"
                            )
                            return True

                        # Check for LangGraph observability attributes
                        attrs = span.get("span_attributes", {})
                        if isinstance(attrs, dict):
                            attr_keys = list(attrs.keys())
                            logger.warning(
                                f"LANGGRAPH_DEBUG: Span {i} attribute keys: {attr_keys[:5]}"
                            )  # First 5 keys

                            for key in attrs.keys():
                                if (
                                    "graph_node" in key
                                    or "langgraph" in key.lower()
                                ):
                                    logger.warning(
                                        f"LANGGRAPH_DEBUG: Found LangGraph attribute: '{key}'"
                                    )
                                    return True

        logger.warning("LANGGRAPH_DEBUG: No LangGraph indicators found")
        return False

    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """
        Extract plan from LangGraph trace, handling Command structures and graph state.
        """
        # Strategy 1: Look for direct plan fields
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

    def _extract_direct_plan_fields(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract plan from direct field names."""
        plan_fields = ["execution_plan", "plan", "agent_plan", "workflow_plan"]

        for field in plan_fields:
            if field in trace_data:
                logger.info(f"LangGraph plan found in direct field '{field}'")
                return trace_data[field]

        return None

    def _extract_plan_from_graph_state(
        self, trace_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract plan from LangGraph graph node state attributes."""
        if "spans" not in trace_data:
            return None

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            attrs = span.get("span_attributes", {})
            if not isinstance(attrs, dict):
                continue

            # Check LangGraph-specific state attributes
            state_keys = [
                "ai.observability.graph_node.input_state",
                "ai.observability.graph_node.output_state",
                "ai.observability.call.kwargs.input",
                "ai.observability.call.return",
            ]

            for state_key in state_keys:
                if state_key in attrs:
                    plan = self._extract_plan_from_state_value(
                        attrs[state_key], state_key
                    )
                    if plan:
                        return plan

        return None

    def _extract_plan_from_state_value(
        self, state_value: Any, state_key: str
    ) -> Optional[Any]:
        """Extract plan from a state value (string or dict)."""
        if isinstance(state_value, str):
            # Try JSON parsing first
            try:
                parsed_state = json.loads(state_value)
                if isinstance(parsed_state, dict):
                    plan_fields = ["execution_plan", "plan", "agent_plan"]
                    for field in plan_fields:
                        if field in parsed_state:
                            logger.info(
                                f"LangGraph plan found in {state_key}.{field}"
                            )
                            return parsed_state[field]
            except json.JSONDecodeError:
                pass

            # Check for plan-related content in string
            if "plan" in state_value.lower() and len(state_value) > 100:
                logger.info(
                    f"LangGraph plan found in {state_key} string content"
                )
                return state_value

        elif isinstance(state_value, dict):
            plan_fields = ["execution_plan", "plan", "agent_plan"]
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
            # Look for Command(...) patterns with execution_plan
            if "Command(" in value and "execution_plan" in value:
                logger.info(
                    f"LangGraph plan found in {location} Command structure"
                )
                return value
            # Look for any plan-related content
            elif "plan" in value.lower() and len(value) > 100:
                logger.info(f"LangGraph plan found in {location} content")
                return value

        elif isinstance(value, dict):
            # Look for update structures: Command(update={...})
            if "update" in value:
                update_data = value["update"]
                if isinstance(update_data, dict):
                    plan_fields = ["execution_plan", "plan", "agent_plan"]
                    for field in plan_fields:
                        if field in update_data:
                            logger.info(
                                f"LangGraph plan found in {location}.update.{field}"
                            )
                            return update_data[field]

        return None

    def extract_execution_flow(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract execution flow from LangGraph spans."""
        flow = []

        if "spans" not in trace_data:
            return flow

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            span_name = span.get("span_name", "")

            # Skip internal LangGraph spans
            if any(
                skip in span_name.lower()
                for skip in ["debug", "log", "trace", "monitor"]
            ):
                continue

            flow_item = {
                "name": span_name,
                "type": "langgraph_node"
                if "graph" in span_name.lower()
                else "step",
            }

            # Add LangGraph-specific attributes
            attrs = span.get("span_attributes", {})
            if isinstance(attrs, dict):
                if "ai.observability.graph_node.latest_message" in attrs:
                    message = attrs[
                        "ai.observability.graph_node.latest_message"
                    ]
                    if isinstance(message, str) and len(message) < 200:
                        flow_item["message"] = message

            flow.append(flow_item)

        return flow

    def extract_agent_interactions(
        self, trace_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract agent interactions from LangGraph traces."""
        interactions = []

        if "spans" not in trace_data:
            return interactions

        for span in trace_data["spans"]:
            if not isinstance(span, dict):
                continue

            attrs = span.get("span_attributes", {})
            if not isinstance(attrs, dict):
                continue

            # Look for LangGraph message attributes
            if "ai.observability.graph_node.latest_message" in attrs:
                message = attrs["ai.observability.graph_node.latest_message"]
                if isinstance(message, str):
                    interaction = {
                        "type": "langgraph_message",
                        "node": span.get("span_name", "unknown"),
                        "content": message[:300],  # Limit content
                    }
                    interactions.append(interaction)

        return interactions


def register_langgraph_provider():
    """Register the LangGraph trace provider."""
    from trulens.core.utils.trace_provider import register_trace_provider

    provider = LangGraphTraceProvider()
    register_trace_provider(provider)
    logger.warning(
        "PROVIDER_DEBUG: LangGraph trace provider registered successfully"
    )


# Auto-register when module is imported
register_langgraph_provider()
