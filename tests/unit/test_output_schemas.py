"""
Tests for feedback output schemas to verify Databricks compatibility.

These tests ensure that Pydantic models used for structured outputs include
`additionalProperties: false` in their JSON schema, which is required by
Databricks and other providers.

See: https://github.com/truera/trulens/issues/2307
"""

import pytest


def test_base_feedback_response_schema_has_additional_properties_false():
    """Verify BaseFeedbackResponse schema includes additionalProperties: false."""
    from trulens.feedback.output_schemas import BaseFeedbackResponse

    schema = BaseFeedbackResponse.model_json_schema()
    assert (
        schema.get("additionalProperties") is False
    ), "BaseFeedbackResponse schema must include 'additionalProperties: false' for Databricks compatibility"


def test_chain_of_thought_response_schema_has_additional_properties_false():
    """Verify ChainOfThoughtResponse schema includes additionalProperties: false."""
    from trulens.feedback.output_schemas import ChainOfThoughtResponse

    schema = ChainOfThoughtResponse.model_json_schema()
    assert (
        schema.get("additionalProperties") is False
    ), "ChainOfThoughtResponse schema must include 'additionalProperties: false' for Databricks compatibility"


def test_base_feedback_response_rejects_extra_fields():
    """Verify BaseFeedbackResponse rejects extra fields at runtime."""
    from pydantic import ValidationError
    from trulens.feedback.output_schemas import BaseFeedbackResponse

    # Valid input should work
    response = BaseFeedbackResponse(score=5)
    assert response.score == 5

    # Extra fields should raise ValidationError
    with pytest.raises(ValidationError):
        BaseFeedbackResponse(score=5, extra_field="not allowed")


def test_chain_of_thought_response_rejects_extra_fields():
    """Verify ChainOfThoughtResponse rejects extra fields at runtime."""
    from pydantic import ValidationError
    from trulens.feedback.output_schemas import ChainOfThoughtResponse

    # Valid input should work
    response = ChainOfThoughtResponse(
        criteria="test criteria",
        supporting_evidence="test evidence",
        score=5,
    )
    assert response.score == 5
    assert response.criteria == "test criteria"

    # Extra fields should raise ValidationError
    with pytest.raises(ValidationError):
        ChainOfThoughtResponse(
            criteria="test",
            supporting_evidence="test",
            score=5,
            extra_field="not allowed",
        )


def test_schema_required_fields():
    """Verify required fields are correctly specified in schemas."""
    from trulens.feedback.output_schemas import BaseFeedbackResponse
    from trulens.feedback.output_schemas import ChainOfThoughtResponse

    base_schema = BaseFeedbackResponse.model_json_schema()
    assert "score" in base_schema.get("required", [])

    cot_schema = ChainOfThoughtResponse.model_json_schema()
    required = cot_schema.get("required", [])
    assert "score" in required
    assert "criteria" in required
    assert "supporting_evidence" in required
