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
        logger.info(f"DEBUG: {type(self).__name__}.compress_trace called")
        logger.info(f"DEBUG: Input trace_data keys: {list(trace_data.keys())}")
        logger.info(
            f"DEBUG: Input trace_data size: {len(json.dumps(trace_data, default=str))} characters"
        )

        compressed = {}

        # Always preserve plan first
        logger.info("DEBUG: Extracting plan")
        plan = self.extract_plan(trace_data)
        logger.info(f"DEBUG: Extracted plan: {plan is not None}")
        if plan:
            compressed["plan"] = plan
            logger.info(
                f"DEBUG: Plan preserved, size: {len(json.dumps(plan, default=str))} characters"
            )
            logger.info("Plan preserved completely for metrics evaluation")
        else:
            logger.warning(
                "No plan found in trace data - this may impact metrics evaluation"
            )

        # Add execution flow
        logger.info("DEBUG: Extracting execution flow")
        flow = self.extract_execution_flow(trace_data)
        logger.info(f"DEBUG: Extracted flow items: {len(flow) if flow else 0}")
        if flow:
            compressed["execution_flow"] = flow

        # Add agent interactions
        logger.info("DEBUG: Extracting agent interactions")
        interactions = self.extract_agent_interactions(trace_data)
        logger.info(
            f"DEBUG: Extracted interactions: {len(interactions) if interactions else 0}"
        )
        if interactions:
            compressed["agent_interactions"] = interactions

        # Add basic trace info
        if "trace_id" in trace_data:
            compressed["trace_id"] = trace_data["trace_id"]
            logger.info("DEBUG: Added trace_id")
        if "metadata" in trace_data:
            compressed["metadata"] = trace_data["metadata"]
            logger.info("DEBUG: Added metadata")

        logger.info(f"DEBUG: Final compressed keys: {list(compressed.keys())}")
        logger.info(
            f"DEBUG: Final compressed size: {len(json.dumps(compressed, default=str))} characters"
        )
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

        logger.info(
            f"DEBUG: {type(self).__name__}.compress_with_plan_priority called"
        )
        logger.info(f"DEBUG: target_token_limit: {target_token_limit}")

        # First, try normal compression but preserve plan
        logger.info("DEBUG: Calling compress_trace for initial compression")
        compressed = self.compress_trace(trace_data)

        # Estimate tokens (rough approximation)
        estimated_tokens = len(json.dumps(compressed, default=str)) // 4
        logger.info(
            f"DEBUG: Initial compressed size: {estimated_tokens} tokens"
        )

        if estimated_tokens <= target_token_limit:
            logger.info("DEBUG: Trace fits within token limit, returning as-is")
            return compressed

        # If still too large, compress non-plan data more aggressively
        logger.warning(
            f"Trace ({estimated_tokens} tokens) exceeds limit ({target_token_limit}), "
            f"applying aggressive compression to non-plan data"
        )

        # Extract plan first
        plan = compressed.get("plan")
        plan_tokens = len(json.dumps(plan, default=str)) // 4 if plan else 0
        logger.info(f"DEBUG: Plan tokens: {plan_tokens}")

        # Rebuild with more aggressive compression for non-plan data
        result = {}
        if plan:
            result["plan"] = plan  # Always preserve plan
            logger.info("DEBUG: Plan preserved in aggressive compression")

        # Add other data within budget
        used_tokens = plan_tokens
        logger.info(f"DEBUG: Starting with {used_tokens} tokens used")

        for key, value in compressed.items():
            if key == "plan":
                continue

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
                            logger.info(
                                f"DEBUG: Added truncated {key} ({i} items), now using {used_tokens} tokens"
                            )
                            break
                    else:
                        logger.info(
                            f"DEBUG: Could not fit {key} even truncated"
                        )
                else:
                    logger.info(
                        f"DEBUG: {key} is not a list or too small to truncate"
                    )

        final_tokens = len(json.dumps(result, default=str)) // 4
        logger.info(
            f"Final compressed trace: {final_tokens} tokens (plan: {plan_tokens})"
        )
        logger.info(f"DEBUG: Final result keys: {list(result.keys())}")

        return result


class GenericTraceProvider(TraceProvider):
    """Generic trace provider for standard trace formats."""

    def can_handle(self, trace_data: Dict[str, Any]) -> bool:
        """Generic provider handles any trace data as fallback."""
        return True

    def extract_plan(self, trace_data: Dict[str, Any]) -> Optional[Any]:
        """Extract plan using generic field names."""
        logger.info("DEBUG: GenericTraceProvider.extract_plan called")
        logger.warning(
            f"PLAN_DEBUG: Available trace_data keys: {list(trace_data.keys())}"
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
        plan_fields = ["plan", "execution_plan", "agent_plan", "workflow_plan"]
        logger.info(f"DEBUG: Checking plan fields: {plan_fields}")

        for field in plan_fields:
            if field in trace_data:
                plan_value = trace_data[field]
                logger.warning(
                    f"PLAN_DEBUG: Plan found in generic field '{field}', type: {type(plan_value)}, size: {len(str(plan_value))}"
                )
                return plan_value
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
                                        logger.warning(
                                            f"PLAN_DEBUG: Found potential plan in span {i} attr '{attr_key}': {type(attr_value)}, size: {len(str(attr_value))}"
                                        )

        logger.warning(
            "PLAN_DEBUG: No plan found in any generic field or spans"
        )
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
        logger.info("DEBUG: TraceProviderRegistry.get_provider called")
        logger.warning(
            f"PROVIDER_DEBUG: Available providers: {[type(p).__name__ for p in self._providers]}"
        )

        for i, provider in enumerate(self._providers):
            provider_name = type(provider).__name__
            logger.warning(
                f"PROVIDER_DEBUG: Checking provider {i}: {provider_name}"
            )

            can_handle = provider.can_handle(trace_data)
            logger.warning(
                f"PROVIDER_DEBUG: {provider_name}.can_handle() returned: {can_handle}"
            )

            if can_handle:
                logger.warning(
                    f"PROVIDER_DEBUG: Selected provider: {provider_name}"
                )
                return provider

        # This should never happen since GenericTraceProvider handles everything
        logger.warning(
            "PROVIDER_DEBUG: No provider could handle trace data, using fallback"
        )
        fallback = self._providers[-1]
        logger.warning(
            f"PROVIDER_DEBUG: Fallback provider: {type(fallback).__name__}"
        )
        return fallback


# Global registry instance
_trace_provider_registry = TraceProviderRegistry()


def register_trace_provider(provider: TraceProvider):
    """Register a trace provider globally."""
    _trace_provider_registry.register_provider(provider)


def get_trace_provider(trace_data: Dict[str, Any]) -> TraceProvider:
    """Get appropriate trace provider for the given data."""
    return _trace_provider_registry.get_provider(trace_data)
