"""
Provider interface for trace-specific parsing and plan extraction.

This allows different app integrations (LangGraph, LangChain, etc.) to provide
their own trace parsing logic while keeping the core compression generic.
"""

from abc import ABC
from abc import abstractmethod
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

        # Always preserve plan first
        plan = self.extract_plan(trace_data)
        if plan:
            compressed["plan"] = plan
            logger.info("Plan preserved completely for metrics evaluation")
        else:
            logger.warning(
                "No plan found in trace data - this may impact metrics evaluation"
            )

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
        import json

        # First, try normal compression but preserve plan
        compressed = self.compress_trace(trace_data)

        # Estimate tokens (rough approximation)
        estimated_tokens = len(json.dumps(compressed, default=str)) // 4

        if estimated_tokens <= target_token_limit:
            return compressed

        # If still too large, compress non-plan data more aggressively
        logger.warning(
            f"Trace ({estimated_tokens} tokens) exceeds limit ({target_token_limit}), "
            f"applying aggressive compression to non-plan data"
        )

        # Extract plan first
        plan = compressed.get("plan")
        plan_tokens = len(json.dumps(plan, default=str)) // 4 if plan else 0

        # Calculate budget for other data

        # Rebuild with more aggressive compression for non-plan data
        result = {}
        if plan:
            result["plan"] = plan  # Always preserve plan

        # Add other data within budget
        used_tokens = plan_tokens
        for key, value in compressed.items():
            if key == "plan":
                continue

            value_tokens = len(json.dumps(value, default=str)) // 4
            if used_tokens + value_tokens <= target_token_limit:
                result[key] = value
                used_tokens += value_tokens
            else:
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
        logger.info(
            f"Final compressed trace: {final_tokens} tokens (plan: {plan_tokens})"
        )

        return result


class GenericTraceProvider(TraceProvider):
    """Generic trace provider for standard trace formats."""

    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """Generic provider handles any trace data as fallback."""
        return True

    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """Extract plan using generic field names."""
        # Check common plan field names
        plan_fields = ["plan", "execution_plan", "agent_plan", "workflow_plan"]

        for field in plan_fields:
            if field in trace_data:
                logger.info(f"Plan found in generic field '{field}'")
                return trace_data[field]

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
        for provider in self._providers:
            if provider.can_handle(trace_data):
                return provider

        # This should never happen since GenericTraceProvider handles everything
        return self._providers[-1]


# Global registry instance
_trace_provider_registry = TraceProviderRegistry()


def register_trace_provider(provider: TraceProvider):
    """Register a trace provider globally."""
    _trace_provider_registry.register_provider(provider)


def get_trace_provider(trace_data: Dict[str, Any]) -> TraceProvider:
    """Get appropriate trace provider for the given data."""
    return _trace_provider_registry.get_provider(trace_data)
