"""Tests for Metric and Feedback argument binding and precedence.

Ensures that Metric.with_arguments, constructor fields, and call-site kwargs
follow hierarchical precedence without silently discarding min_score_val,
max_score_val, or temperature.
"""

from typing import Any, Dict, List, Optional

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.core.metric import metric as core_metric


def _mock_implementation(
    prompt: str = "default_prompt",
    min_score_val: int = 0,
    max_score_val: int = 3,
    temperature: float = 0.0,
    criteria: Optional[str] = None,
    examples: Optional[List] = None,
    additional_instructions: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Mock evaluation function that returns all received arguments."""
    return {
        "prompt": prompt,
        "min_score_val": min_score_val,
        "max_score_val": max_score_val,
        "temperature": temperature,
        "criteria": criteria,
        "examples": examples,
        "additional_instructions": additional_instructions,
        "kwargs": kwargs,
    }


def test_with_arguments_binds_score_range_and_temperature() -> None:
    """Test with_arguments binds min_score_val, max_score_val, temperature."""
    metric = core_metric.Metric(
        implementation=_mock_implementation
    ).with_arguments(
        min_score_val=1,
        max_score_val=10,
        temperature=0.8,
        criteria="strict",
    )

    result = metric()
    assert result["min_score_val"] == 1
    assert result["max_score_val"] == 10
    assert result["temperature"] == 0.8
    assert result["criteria"] == "strict"


def test_constructor_arguments_used_when_unoverridden() -> None:
    """Test that explicit constructor arguments are passed when not overridden."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        min_score_val=2,
        max_score_val=7,
        temperature=0.4,
        criteria="rubric",
    )

    result = metric()
    assert result["min_score_val"] == 2
    assert result["max_score_val"] == 7
    assert result["temperature"] == 0.4
    assert result["criteria"] == "rubric"


def test_with_arguments_overrides_constructor_arguments() -> None:
    """Test with_arguments takes precedence over Metric constructor fields."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        min_score_val=0,
        max_score_val=5,
        temperature=0.1,
    ).with_arguments(
        max_score_val=12,
        temperature=0.9,
    )

    result = metric()
    assert result["min_score_val"] == 0
    assert result["max_score_val"] == 12
    assert result["temperature"] == 0.9


def test_call_site_kwargs_override_constructor_arguments() -> None:
    """Test call-site kwargs override Metric constructor fields."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        max_score_val=5,
        temperature=0.2,
    )

    result = metric(max_score_val=20, temperature=0.7)
    assert result["max_score_val"] == 20
    assert result["temperature"] == 0.7


def test_call_site_and_bound_overlap_raises_value_error() -> None:
    """Test that providing an argument both bound and at call-site raises ValueError."""
    metric = core_metric.Metric(
        implementation=_mock_implementation
    ).with_arguments(max_score_val=10)

    with pytest.raises(
        ValueError, match="Metric arguments cannot be both selected and bound"
    ):
        metric(max_score_val=15)


def test_implementation_native_defaults_preserved_when_unspecified() -> None:
    """Test that underlying implementation defaults are honored when not set."""

    def custom_attribution_fn(
        statement: str = "test",
        min_score_val: int = 0,
        max_score_val: int = 1,
        temperature: float = 0.5,
    ) -> Dict[str, Any]:
        return {
            "statement": statement,
            "min_score_val": min_score_val,
            "max_score_val": max_score_val,
            "temperature": temperature,
        }

    metric = core_metric.Metric(implementation=custom_attribution_fn)
    result = metric()
    # Should preserve the implementation's native max_score_val=1 and temperature=0.5
    assert result["min_score_val"] == 0
    assert result["max_score_val"] == 1
    assert result["temperature"] == 0.5


def test_feedback_subclass_precedence() -> None:
    """Test that deprecated Feedback class also respects argument precedence."""
    with pytest.deprecated_call():
        fb = core_feedback.Feedback(
            _mock_implementation,
            max_score_val=4,
        ).with_arguments(
            max_score_val=8,
            temperature=0.6,
        )

    result = fb()
    assert result["max_score_val"] == 8
    assert result["temperature"] == 0.6
