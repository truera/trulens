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
        self.providers = {}
        self._register_default_providers()

    def _register_default_providers(self):
        """Register default trace providers."""
        # Import providers here to avoid circular imports
        try:
            from trulens.core.utils.trace_provider import DefaultTraceProvider

            self.providers["default"] = DefaultTraceProvider()
        except ImportError:
            logger.warning("Default trace provider not available")

        try:
            from trulens.apps.langgraph.trace_provider import (
                LangGraphTraceProvider,
            )

            self.providers["langgraph"] = LangGraphTraceProvider()
        except ImportError:
            logger.debug("LangGraph trace provider not available")

    def register_provider(self, name: str, provider):
        """Register a custom trace provider."""
        self.providers[name] = provider

    def _detect_trace_type(self, data: Any) -> str:
        """Detect the type of trace data."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return "default"

        if isinstance(data, dict):
            # Check for LangGraph indicators
            if "spans" in data:
                for span in data.get("spans", []):
                    if isinstance(span, dict):
                        attrs = span.get("span_attributes", {})
                        if any(
                            "langgraph" in str(key).lower()
                            for key in attrs.keys()
                        ):
                            return "langgraph"

        return "default"

    def compress_trace_with_plan_priority(
        self, trace_data: Any, target_token_limit: int = 100000
    ) -> Dict[str, Any]:
        """
        Compress trace with plan preservation as highest priority.
        If context window is exceeded, compress other data more aggressively.
        """
        # Detect trace type and use appropriate provider
        trace_type = self._detect_trace_type(trace_data)
        provider = self.providers.get(trace_type, self.providers.get("default"))

        if provider:
            return provider.compress_with_plan_priority(
                trace_data, target_token_limit
            )

        # Fallback to basic compression
        return self.compress_trace(trace_data)

    def compress_trace(self, trace_data: Any) -> Dict[str, Any]:
        """
        Compress trace data using the appropriate provider.

        Args:
            trace_data: The trace data to compress

        Returns:
            Compressed trace data
        """
        # Convert string to dict if needed
        if isinstance(trace_data, str):
            try:
                data = json.loads(trace_data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse trace data as JSON")
                return {"error": "Invalid JSON trace data"}
        else:
            data = trace_data

        if not isinstance(data, dict):
            logger.warning("Trace data is not a dictionary")
            return {"error": "Trace data must be a dictionary"}

        logger.info(
            "PLAN_PRESERVATION_DEBUG: Using modified trace compression with plan preservation"
        )

        # Detect trace type and use appropriate provider
        trace_type = self._detect_trace_type(data)
        provider = self.providers.get(trace_type, self.providers.get("default"))

        if provider:
            return provider.compress_trace(data)

        # Fallback compression if no provider available
        return self._basic_compression(data)

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
    compressor = TraceCompressor()

    if preserve_plan:
        return compressor.compress_trace_with_plan_priority(
            trace_data, target_token_limit
        )
    else:
        return compressor.compress_trace(trace_data)
