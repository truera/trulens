from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class BaseFeedbackResponse(BaseModel):
    """
    A base model for feedback responses.
    It can be extended to include specific fields for different feedback types.

    Note: The `extra="forbid"` config ensures that `additionalProperties: false`
    is included in the JSON schema, which is required by providers like Databricks
    for structured outputs.
    """

    model_config = ConfigDict(extra="forbid")

    score: int = Field(description="The score based on the given criteria.")


class ChainOfThoughtResponse(BaseModel):
    """
    A model to represent the response from a Chain of Thought (COT) evaluation.
    It includes the criteria, supporting evidence, and score.

    Note: The `extra="forbid"` config ensures that `additionalProperties: false`
    is included in the JSON schema, which is required by providers like Databricks
    for structured outputs.
    """

    model_config = ConfigDict(extra="forbid")

    criteria: str = Field(description="The criteria for the evaluation.")
    supporting_evidence: str = Field(
        description="Supporting evidence for the score, detailing the reasoning step by step."
    )
    score: int = Field(description="The score based on the given criteria.")


class RequirementEvaluation(BaseModel):
    """A verdict and evidence for one requirement."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(description="The requirement being evaluated.")
    verdict: Literal["met", "partly_met", "not_met"] = Field(
        description="Whether the assistant met all, part, or none of the requirement."
    )
    evidence: str = Field(
        description="Concise evidence from the assistant output for the verdict."
    )


class RequirementExtractionResponse(BaseModel):
    """Requirements extracted from a request without seeing an output."""

    model_config = ConfigDict(extra="forbid")

    requirements: list[str] = Field(
        description="Distinct requirements stated by the user, in order."
    )


class RequirementSatisfactionResponse(BaseModel):
    """Structured requirement-by-requirement evaluation output."""

    model_config = ConfigDict(extra="forbid")

    evaluations: list[RequirementEvaluation] = Field(
        description="One ordered evaluation for each requirement."
    )
