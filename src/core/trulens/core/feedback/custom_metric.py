"""Client-side custom metrics functionality."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.selector import Selector


class EvaluationConfig:
    """
    Configuration for mapping metric function parameters to OTEL span attributes.

    This class provides a structured way to define how metric function arguments
    should be extracted from OTEL spans during evaluation.
    """

    def __init__(
        self,
        name: str,
        metric_type: str = "custom",
        computation_type: str = "client",
        selectors: Optional[Dict[str, Selector]] = None,
        higher_is_better: bool = True,
        description: Optional[str] = None,
    ):
        """
        Initialize an evaluation configuration.

        Args:
            name: Name of the metric configuration
            metric_type: Unique identifier for the metric (e.g., "text2SQL", "accuracy"). equivalent to metric name
            computation_type: Where to compute ("client" or "server")
            selectors: Dictionary mapping parameter names to Selectors
            higher_is_better: Whether higher scores are better
            description: Optional description of the configuration
        """
        self.name = name
        self.metric_type = metric_type
        self.computation_type = computation_type
        self.selectors = selectors or {}
        self.higher_is_better = higher_is_better
        self.description = description

    def add_selector(
        self, parameter_name: str, selector: Selector
    ) -> "EvaluationConfig":
        """
        Add a selector for a specific parameter.

        Args:
            parameter_name: Name of the metric function parameter
            selector: Selector that extracts the value from spans

        Returns:
            Self for method chaining
        """
        self.selectors[parameter_name] = selector
        return self

    def validate_for_function(self, func: Callable) -> None:
        """
        Validate that this configuration matches the given function signature.

        Args:
            func: The metric function to validate against

        Raises:
            ValueError: If selectors don't match function parameters
        """
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())
        selector_params = set(self.selectors.keys())

        if func_params != selector_params:
            missing = func_params - selector_params
            extra = selector_params - func_params
            error_msg = f"Evaluation config '{self.name}' doesn't match function parameters"
            if missing:
                error_msg += f"\nMissing selectors: {missing}"
            if extra:
                error_msg += f"\nExtra selectors: {extra}"
            raise ValueError(error_msg)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        """
        Create an EvaluationConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration data

        Returns:
            EvaluationConfig instance
        """
        return cls(
            name=config_dict["name"],
            metric_type=config_dict.get("metric_type", "custom"),
            computation_type=config_dict.get("computation_type", "client"),
            selectors=config_dict.get("selectors", {}),
            higher_is_better=config_dict.get("higher_is_better", True),
            description=config_dict.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation config to a dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "name": self.name,
            "metric_type": self.metric_type,
            "computation_type": self.computation_type,
            "selectors": self.selectors,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationConfig(name='{self.name}', type='{self.metric_type}', "
            f"computation='{self.computation_type}', selectors={len(self.selectors)})"
        )


def custom_metric(
    name: Optional[str] = None,
    higher_is_better: bool = True,
    metric_type: str = "custom",
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator to convert a function into a client-side custom metric.

    This decorator transforms a regular function into a CustomMetric instance
    that can be registered with TruApp for client-side evaluation.

    Args:
        name: Name of the metric (defaults to function name)
        higher_is_better: Whether higher scores are better for this metric
        metric_type: Type identifier for the metric (e.g., "text2SQL", "accuracy")
        description: Optional description of what the metric measures

    Returns:
        A decorator function that wraps the target function as a CustomMetric

    Example:
        ```python
        @custom_metric(name="text2sql_accuracy", higher_is_better=True)
        def text_to_sql_scores(query: str, sql: str) -> float:
            if "SELECT" in sql.upper() and "movies" in query.lower():
                return 0.9
            return 0.1
        ```
    """

    def decorator(func: Callable) -> CustomMetric:
        metric_name = name or func.__name__
        return CustomMetric(
            function=func,
            metric_type=metric_name,
            higher_is_better=higher_is_better,
            description=description,
        )

    return decorator


class CustomMetric:
    """
    Wrapper class for custom metric functions.

    This class encapsulates a user-defined metric function along with its
    metadata and provides methods to integrate with the TruLens feedback system.
    """

    def __init__(
        self,
        function: Callable,
        metric_type: str,
        higher_is_better: bool = True,
        description: Optional[str] = None,
    ):
        """
        Initialize a CustomMetric.

        Args:
            function: The metric function to wrap
            metric_type: identifier for the metric. equivalent to metric name
            higher_is_better: Whether higher scores are better
            description: Optional description
        """
        self.function = function
        self.higher_is_better = higher_is_better
        self.metric_type = metric_type
        self.description = description or f"Custom metric: {metric_type}"

        # Validate function signature
        self._validate_function()

    def _validate_function(self) -> None:
        """Validate that the function has a proper signature for a metric."""
        sig = inspect.signature(self.function)

        # Check that function has at least one parameter
        if len(sig.parameters) == 0:
            raise ValueError(
                f"Metric function '{self.metric_type}' must have at least one parameter"
            )

        # Check return annotation if present
        if sig.return_annotation != inspect.Signature.empty:
            expected_types = (float, int, tuple)
            if not any(
                sig.return_annotation == t
                or (
                    hasattr(sig.return_annotation, "__origin__")
                    and sig.return_annotation.__origin__ in (tuple, Union)
                )
                for t in expected_types
            ):
                # This is just a warning, not an error
                pass

    def __call__(
        self, *args, **kwargs
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """Call the underlying metric function."""
        return self.function(*args, **kwargs)

    def create_feedback_definition(
        self, selectors: Dict[str, Selector]
    ) -> Feedback:
        """
        Create a Feedback instance from this custom metric.

        Args:
            selectors: Dictionary mapping parameter names to Selectors

        Returns:
            A Feedback instance configured for this custom metric

        Raises:
            ValueError: If selectors don't match function parameters
        """
        # Validate that selectors match function parameters
        sig = inspect.signature(self.function)
        func_params = set(sig.parameters.keys())
        selector_params = set(selectors.keys())

        if func_params != selector_params:
            missing = func_params - selector_params
            extra = selector_params - func_params
            error_msg = f"Selector parameters don't match function parameters for '{self.metric_type}'"
            if missing:
                error_msg += f"\nMissing selectors: {missing}"
            if extra:
                error_msg += f"\nExtra selectors: {extra}"
            raise ValueError(error_msg)

        return Feedback(
            imp=self.function,
            name=self.metric_type,
            higher_is_better=self.higher_is_better,
        ).on(selectors)

    def create_feedback_from_config(self, config: EvaluationConfig) -> Feedback:
        """
        Create a Feedback instance from an EvaluationConfig.

        Args:
            config: EvaluationConfig containing selectors and metadata

        Returns:
            A Feedback instance configured for this custom metric

        Raises:
            ValueError: If config doesn't match function parameters
        """
        # Validate the config against this function
        config.validate_for_function(self.function)

        return Feedback(
            imp=self.function,
            name=config.name,
            higher_is_better=config.higher_is_better,
        ).on(config.selectors)

    def __repr__(self) -> str:
        return (
            f"CustomMetric(name='{self.metric_type}', "
            f"higher_is_better={self.higher_is_better})"
        )
