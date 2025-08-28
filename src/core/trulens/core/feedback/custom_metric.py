"""Client-side custom metrics functionality."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.selector import Selector


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
            name=metric_name,
            higher_is_better=higher_is_better,
            metric_type=metric_type,
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
        name: str,
        higher_is_better: bool = True,
        metric_type: str = "custom",
        description: Optional[str] = None,
    ):
        """
        Initialize a CustomMetric.

        Args:
            function: The metric function to wrap
            name: Name of the metric
            higher_is_better: Whether higher scores are better
            metric_type: Type identifier for the metric
            description: Optional description
        """
        self.function = function
        self.name = name
        self.higher_is_better = higher_is_better
        self.metric_type = metric_type
        self.description = description or f"Custom metric: {name}"

        # Validate function signature
        self._validate_function()

    def _validate_function(self) -> None:
        """Validate that the function has a proper signature for a metric."""
        sig = inspect.signature(self.function)

        # Check that function has at least one parameter
        if len(sig.parameters) == 0:
            raise ValueError(
                f"Metric function '{self.name}' must have at least one parameter"
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
            error_msg = f"Selector parameters don't match function parameters for '{self.name}'"
            if missing:
                error_msg += f"\nMissing selectors: {missing}"
            if extra:
                error_msg += f"\nExtra selectors: {extra}"
            raise ValueError(error_msg)

        return Feedback(
            implementation=self.function,
            name=self.name,
            higher_is_better=self.higher_is_better,
        ).on(selectors)

    def __repr__(self) -> str:
        return (
            f"CustomMetric(name='{self.name}', type='{self.metric_type}', "
            f"higher_is_better={self.higher_is_better})"
        )
