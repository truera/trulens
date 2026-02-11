"""
DEPRECATED: This module is deprecated. Use trulens.core.metric instead.

The Feedback class is now a deprecated alias for Metric.
This module is maintained for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)
import warnings

from trulens.core.metric.metric import GroundednessConfigs
from trulens.core.metric.metric import InvalidSelector
from trulens.core.metric.metric import Metric
from trulens.core.metric.metric import SkipEval
from trulens.core.schema import feedback as feedback_schema

logger = logging.getLogger(__name__)

# Re-export these for backward compatibility
__all__ = [
    "Feedback",
    "SnowflakeFeedback",
    "SkipEval",
    "InvalidSelector",
    "GroundednessConfigs",
]


class Feedback(Metric):
    """
    DEPRECATED: Use Metric instead.

    This class is maintained for backward compatibility. All functionality
    has been moved to the Metric class.

    Example of migrating to Metric:
        ```python
        # Old way (deprecated):
        from trulens.core import Feedback
        feedback = Feedback(provider.relevance).on_input_output()

        # New way (recommended):
        from trulens.core import Metric
        metric = Metric(implementation=provider.relevance).on_input_output()
        ```
    """

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        examples: Optional[List[Tuple]] = None,
        criteria: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        min_score_val: Optional[int] = 0,
        max_score_val: Optional[int] = 3,
        temperature: Optional[float] = 0.0,
        groundedness_configs: Optional[GroundednessConfigs] = None,
        enable_trace_compression: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize a Feedback (deprecated, use Metric instead).

        Args:
            imp: The feedback function to execute. DEPRECATED: Use `implementation` instead.
            agg: Aggregator function for combining multiple feedback results.
            examples: User-supplied examples for this feedback function.
            criteria: Criteria for the feedback evaluation.
            additional_instructions: Custom instructions for the feedback function.
            min_score_val: Minimum score value (default: 0).
            max_score_val: Maximum score value (default: 3).
            temperature: Temperature parameter for LLM-based feedback (default: 0.0).
            groundedness_configs: Optional groundedness configuration.
            enable_trace_compression: Whether to compress trace data.
            **kwargs: Additional arguments passed to parent class.
        """
        warnings.warn(
            "Feedback is deprecated and will be removed in a future version. "
            "Use Metric instead:\n"
            "  from trulens.core import Metric\n"
            "  metric = Metric(implementation=fn).on_input_output()",
            DeprecationWarning,
            stacklevel=2,
        )

        # Pass imp as implementation to Metric
        # Metric's __init__ handles the 'imp' parameter for backward compatibility
        super().__init__(
            imp=imp,
            agg=agg,
            examples=examples,
            criteria=criteria,
            additional_instructions=additional_instructions,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
            groundedness_configs=groundedness_configs,
            enable_trace_compression=enable_trace_compression,
            **kwargs,
        )


class SnowflakeFeedback(Feedback):
    """[DEPRECATED] Similar to the parent class Feedback except this ensures the feedback is run only on the Snowflake server.

    This class is deprecated and will be removed in the next major release. Please use Metric or [Snowflake AI Observability](https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications) instead.
    """

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        **kwargs,
    ):
        # Note: Feedback.__init__ will emit its own deprecation warning
        # We add an additional one specific to SnowflakeFeedback
        warnings.warn(
            "SnowflakeFeedback is deprecated and will be removed in the next major release. "
            "Please use Metric or Snowflake AI Observability instead: "
            "https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications",
            DeprecationWarning,
            stacklevel=2,
        )
        if (
            not hasattr(imp, "__self__")
            or str(type(imp.__self__))
            != "<class 'trulens.providers.cortex.provider.Cortex'>"
        ):
            raise ValueError(
                "`SnowflakeFeedback` can only support feedback functions defined in "
                "`trulens-providers-cortex` package's `trulens.providers.cortex.provider.Cortex` class!"
            )
        super().__init__(imp, agg, **kwargs)
        self.run_location = feedback_schema.FeedbackRunLocation.SNOWFLAKE


# Rebuild models to ensure proper pydantic serialization
Feedback.model_rebuild()
