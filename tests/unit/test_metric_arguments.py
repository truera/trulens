"""Tests for Metric and Feedback argument binding and precedence.

Ensures that Metric.with_arguments, constructor fields, and call-site kwargs
follow hierarchical precedence without silently discarding min_score_val,
max_score_val, or temperature, and that None consistently defers to the
implementation's native defaults at all layers.
"""

from types import MethodType
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.core.metric import metric as core_metric
from trulens.feedback import llm_provider as feedback_llm_provider


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


def _mock_provider_self() -> MagicMock:
    """Mock LLMProvider instance with real helper methods."""
    mock = MagicMock(spec=feedback_llm_provider.LLMProvider)
    mock._number_citation_sources = (
        feedback_llm_provider.LLMProvider._number_citation_sources
    )
    mock._determine_output_space = MethodType(
        feedback_llm_provider.LLMProvider._determine_output_space, mock
    )
    return mock


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
    """Test explicit constructor arguments are passed when not overridden."""
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
    """Test that providing an argument both bound and at call-site raises."""
    metric = core_metric.Metric(
        implementation=_mock_implementation
    ).with_arguments(max_score_val=10)

    with pytest.raises(
        ValueError, match="Metric arguments cannot be both selected and bound"
    ):
        metric(max_score_val=15)


def test_implementation_native_defaults_preserved_when_unspecified() -> None:
    """Test underlying implementation defaults are honored when not set."""

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
    # Should preserve the implementation's native max_score_val=1 and temp=0.5
    assert result["min_score_val"] == 0
    assert result["max_score_val"] == 1
    assert result["temperature"] == 0.5


def test_none_in_constructor_uses_implementation_default() -> None:
    """Test explicit None in constructor falls back to implementation default."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        min_score_val=None,
        max_score_val=None,
        temperature=None,
    )
    result = metric()
    assert result["min_score_val"] == 0
    assert result["max_score_val"] == 3
    assert result["temperature"] == 0.0


def test_none_in_with_arguments_uses_implementation_default() -> None:
    """Test passing None in with_arguments uses implementation default."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        max_score_val=10,
    ).with_arguments(max_score_val=None)

    result = metric()
    # Bound None overrides constructor 10 and defers to native default 3
    assert result["max_score_val"] == 3


def test_none_at_call_site_uses_implementation_default() -> None:
    """Test passing None at call-site uses implementation default."""
    metric = core_metric.Metric(
        implementation=_mock_implementation,
        max_score_val=10,
    )

    result = metric(max_score_val=None)
    # Call-site None overrides constructor 10 and defers to native default 3
    assert result["max_score_val"] == 3


def test_metric_citation_attribution_end_to_end() -> None:
    """Test wrapping real citation_attribution in Metric preserves max=1."""
    mock = _mock_provider_self()
    mock.citation_attribution = MethodType(
        feedback_llm_provider.LLMProvider.citation_attribution, mock
    )
    mock.generate_score = MagicMock(return_value=1.0)

    metric = core_metric.Metric(implementation=mock.citation_attribution)
    score = metric(
        question="When did it happen?",
        source=["It happened in 1991."],
        statement="It happened in 1991 [1].",
    )
    assert score == 1.0
    mock.generate_score.assert_called_once()
    _, kwargs = mock.generate_score.call_args
    # Binary output space (0 to 1) preserved end-to-end
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 1


def test_metric_citation_attribution_with_arguments_override() -> None:
    """Test with_arguments on citation_attribution can customize score range."""
    mock = _mock_provider_self()
    mock.citation_attribution = MethodType(
        feedback_llm_provider.LLMProvider.citation_attribution, mock
    )
    mock.generate_score = MagicMock(return_value=1.0)

    metric = core_metric.Metric(
        implementation=mock.citation_attribution
    ).with_arguments(max_score_val=3)

    score = metric(
        question="When did it happen?",
        source=["It happened in 1991."],
        statement="It happened in 1991 [1].",
    )
    assert score == 1.0
    mock.generate_score.assert_called_once()
    _, kwargs = mock.generate_score.call_args
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 3


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
