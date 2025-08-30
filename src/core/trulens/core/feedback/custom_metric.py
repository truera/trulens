"""Client-side custom metrics functionality."""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.selector import Selector

# Global registry for automatic metric registration
_metric_registry_lock = threading.Lock()
_pending_metrics: List[Tuple[CustomMetric, Optional[EvaluationConfig]]] = []


def register_pending_metrics_with_app(tru_app) -> int:
    """Register all pending metrics with the given TruApp."""
    global _pending_metrics

    registered_count = 0
    with _metric_registry_lock:
        # Process all pending metrics
        for metric, eval_config in _pending_metrics[:]:
            try:
                if eval_config:
                    # Use the direct method to avoid circular imports
                    _register_metric_directly(tru_app, metric, eval_config)
                else:
                    # Need to create a default evaluation config
                    default_config = EvaluationConfig(
                        metric_type=metric.metric_type,
                        computation_type="client",
                    )
                    # Add default selectors for single-parameter functions
                    sig = inspect.signature(metric.function)
                    if len(sig.parameters) == 1:
                        param_name = list(sig.parameters.keys())[0]
                        # Import SpanAttributes here to avoid circular imports
                        try:
                            from trulens.otel.semconv.trace import (
                                SpanAttributes,
                            )

                            default_config.add_selector(
                                param_name,
                                Selector(
                                    span_type=SpanAttributes.SpanType.RECORD_ROOT,
                                    span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
                                ),
                            )
                            _register_metric_directly(
                                tru_app, metric, default_config
                            )
                        except ImportError:
                            # Skip if can't import SpanAttributes
                            continue
                    else:
                        # For multi-parameter functions, user must provide explicit config
                        continue

                registered_count += 1
                _pending_metrics.remove((metric, eval_config))

            except Exception as e:
                # Log error but continue with other metrics
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to auto-register metric {metric.metric_type}: {e}"
                )

    return registered_count


def _register_metric_directly(
    tru_app, metric: CustomMetric, eval_config: EvaluationConfig
):
    """Register a metric directly with TruApp to avoid circular imports."""
    # Create feedback definition
    feedback_def = metric.create_feedback_from_config(eval_config)

    # Store custom metric info
    custom_metric_info = {
        "metric": metric,
        "feedback": feedback_def,
        "evaluation_config": eval_config,
        "selectors": eval_config.selectors,
        "metric_type": eval_config.metric_type,
        "computation_type": eval_config.computation_type,
    }

    tru_app.custom_metrics.append(custom_metric_info)


def _auto_register_metric(
    metric: CustomMetric, eval_config: Optional[EvaluationConfig] = None
):
    """Add metric to pending registry for later registration with TruApp."""
    global _pending_metrics

    with _metric_registry_lock:
        # Simply add to pending metrics - will be registered when TruApp is created
        _pending_metrics.append((metric, eval_config))
        return True


class EvaluationConfig:
    """
    Configuration for mapping metric function parameters to OTEL span attributes.

    This class provides a structured way to define how metric function arguments
    should be extracted from OTEL spans during evaluation.
    """

    def __init__(
        self,
        metric_type: str = "custom",
        computation_type: str = "client",
        selectors: Optional[Dict[str, Selector]] = None,
        higher_is_better: bool = True,
        description: Optional[str] = None,
    ):
        """
        Initialize an evaluation configuration.

        Args:
            metric_type: Unique identifier for the metric (e.g., "text2SQL", "accuracy"). equivalent to metric name
            computation_type: Where to compute ("client" or "server")
            selectors: Dictionary mapping parameter names to Selectors
            higher_is_better: Whether higher scores are better
            description: Optional description of the configuration
        """
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
            error_msg = f"Evaluation config for metric '{self.metric_type}' doesn't match function parameters"
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
            "metric_type": self.metric_type,
            "computation_type": self.computation_type,
            "selectors": self.selectors,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationConfig(metric_type='{self.metric_type}', "
            f"computation='{self.computation_type}', selectors={len(self.selectors)})"
        )


def custom_metric(
    metric_type: Optional[str] = None,
    higher_is_better: bool = True,
    description: Optional[str] = None,
    evaluation_config: Optional[EvaluationConfig] = None,
    auto_register: bool = True,
) -> Callable:
    """
    Decorator to convert a function into a client-side custom metric.

    This decorator transforms a regular function into a CustomMetric instance
    that can be automatically registered with TruApp for client-side evaluation.

    Args:
        metric_type: identifier for the metric. equivalent to metric name (e.g., "text2SQL", "accuracy")
        higher_is_better: Whether higher scores are better for this metric
        description: Optional description of what the metric measures
        evaluation_config: Optional EvaluationConfig for explicit span-to-argument mapping
        auto_register: Whether to automatically register with TruApp when available

    Returns:
        A decorator function that wraps the target function as a CustomMetric

    Example:
        ```python
        # Basic usage - automatically registers with current TruApp context
        @custom_metric(metric_type="text2sql_accuracy", higher_is_better=True)
        def text_to_sql_scores(query: str, sql: str) -> float:
            if "SELECT" in sql.upper() and "movies" in query.lower():
                return 0.9
            return 0.1

        # With explicit evaluation config
        eval_config = EvaluationConfig(metric_type="custom_accuracy").add_selector(...)
        @custom_metric(evaluation_config=eval_config)
        def custom_accuracy(query: str) -> float:
            return len(query) / 100.0
        ```
    """

    def decorator(func: Callable) -> CustomMetric:
        custom_metric_instance = CustomMetric(
            function=func,
            metric_type=metric_type or func.__name__,
            higher_is_better=higher_is_better,
            description=description,
        )

        # Automatically register with current TruApp context if enabled
        if auto_register:
            _auto_register_metric(custom_metric_instance, evaluation_config)

        return custom_metric_instance

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
            self.function,
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
            self.function,
            name=self.metric_type,  # Use the metric's name, not the eval config name
            higher_is_better=config.higher_is_better,
        ).on(config.selectors)

    def __repr__(self) -> str:
        return (
            f"CustomMetric(name='{self.metric_type}', "
            f"higher_is_better={self.higher_is_better})"
        )
