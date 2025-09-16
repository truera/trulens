"""Client-side custom metrics functionality."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.selector import Selector


class MetricConfig:
    """
    Configuration for a custom metric including implementation and span mapping.

    This class defines a complete metric configuration that includes the metric
    function implementation and how its arguments should be extracted from OTEL spans.

    Key Concepts:
    - metric_name: Unique semantic identifier for this specific usage of the metric
                   (e.g., "text2sql_accuracy_v1", "relevance_for_qa_task")
    - metric_type: Implementation identifier of the underlying metric function
                   (e.g., "text2sql", "accuracy", "relevance")

    This distinction allows the same metric implementation to be used multiple times
    with different configurations and names within the same application.
    """

    def __init__(
        self,
        metric_name: str,
        metric_implementation: Callable,
        metric_type: Optional[str] = None,
        selectors: Optional[Dict[str, Selector]] = None,
        computation_type: str = "client",
        higher_is_better: bool = True,
        description: Optional[str] = None,
    ):
        """
        Initialize a metric configuration.

        Args:
            metric_name: Unique semantic identifier for this specific metric usage
                        (e.g., "text2sql_accuracy_v1", "custom_relevance_for_qa")
            metric_implementation: The metric function to execute
            metric_type: Implementation identifier of the custom metric
                        (e.g., "text2sql", "accuracy", "relevance").
                        If not provided, defaults to the function name.
            selectors: Dictionary mapping parameter names to Selectors
            computation_type: Where to compute ("client" or "server")
            higher_is_better: Whether higher scores are better
            description: Optional description of the metric
        """
        self.metric_name = metric_name
        self.metric_implementation = metric_implementation
        self.metric_type = metric_type or (
            metric_implementation.__name__
            if hasattr(metric_implementation, "__name__")
            else "custom_metric"
        )
        self.computation_type = computation_type
        self.selectors = selectors or {}
        self.higher_is_better = higher_is_better
        self.description = description

        # Validate function signature
        self._validate_function()

    def _validate_function(self) -> None:
        """Validate that the function has a proper signature for a metric."""
        if not callable(self.metric_implementation):
            raise ValueError("metric_implementation must be callable")

        sig = inspect.signature(self.metric_implementation)

        # Check that function has at least one parameter
        if len(sig.parameters) == 0:
            raise ValueError(
                f"Metric function '{self.metric_name}' must have at least one parameter"
            )

    def validate_selectors(self) -> None:
        """
        Validate that selectors match the function signature.

        Only checks required parameters (those without default values).

        Raises:
            ValueError: If selectors don't match function parameters
        """
        sig = inspect.signature(self.metric_implementation)

        # Get all function parameters
        function_args = list(sig.parameters.keys())

        # Get only required parameters (those without defaults)
        required_function_args = [
            param_name
            for param_name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
        ]

        error_msg = ""

        # Check for extra selectors
        extra_selectors = []
        for selector in self.selectors:
            if selector not in function_args:
                extra_selectors.append(selector)
        if extra_selectors:
            error_msg += f"Metric function '{self.metric_name}' has selectors that are not in the function signature:\n"
            error_msg += f"Extra selectors: {extra_selectors}\n"
            error_msg += f"Function args: {function_args}\n"

        # Check for missing required selectors (only required parameters)
        missing_selectors = []
        for required_function_arg in required_function_args:
            if required_function_arg not in self.selectors:
                missing_selectors.append(required_function_arg)
        if missing_selectors:
            error_msg += (
                f"Metric function '{self.metric_name}' has missing selectors:\n"
            )
            error_msg += f"Missing selectors: {missing_selectors}\n"
            error_msg += f"Required function args: {required_function_args}\n"

        if error_msg:
            raise ValueError(error_msg)

    def create_feedback_definition(self) -> Feedback:
        """
        Create a Feedback instance from this metric configuration.

        Returns:
            A Feedback instance configured for this metric

        Raises:
            ValueError: If selectors don't match function parameters
        """
        self.validate_selectors()

        # For client-side metrics, we don't need serialization since they won't be deferred
        # Create a non-serializable implementation placeholder to avoid serialization warnings
        from trulens.core.utils import pyschema as pyschema_utils

        try:
            # Try to create a serializable implementation
            implementation = pyschema_utils.FunctionOrMethod.of_callable(
                self.metric_implementation, loadable=True
            )
        except Exception:
            # If serialization fails (e.g., function in __main__, like client side metrics defined in a notebook), create non-loadable version
            # This is fine for client-side metrics that execute immediately
            implementation = pyschema_utils.FunctionOrMethod.of_callable(
                self.metric_implementation, loadable=False
            )

        return Feedback(
            imp=self.metric_implementation,
            implementation=implementation,  # Pre-provide serialized version
            name=self.metric_name,
            higher_is_better=self.higher_is_better,
        ).on(self.selectors)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metric config to a dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "computation_type": self.computation_type,
            "selectors": self.selectors,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"MetricConfig(metric_name='{self.metric_name}', "
            f"metric_type='{self.metric_type}', "
            f"computation='{self.computation_type}', selectors={len(self.selectors)})"
        )
