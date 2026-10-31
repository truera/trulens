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


class TurnVerdict(BaseModel):
    """The judge's verdict on one assistant turn."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(
        description="The turn number, as labelled in the transcript."
    )
    adheres: bool = Field(
        description="True if the assistant stayed in the role on this turn."
    )
    reason: str = Field(
        description="One sentence on why the turn adheres or breaks the role."
    )


class RoleAdherenceResponse(BaseModel):
    """A per-turn verdict on whether the assistant kept to its role.

    Note: The `extra="forbid"` config ensures that `additionalProperties: false`
    is included in the JSON schema, which is required by providers like Databricks
    for structured outputs.
    """

    model_config = ConfigDict(extra="forbid")

    turns: list[TurnVerdict] = Field(
        description="One verdict per assistant turn, in transcript order."
    )


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


class EscalationTrigger(BaseModel):
    """A turn where the escalation policy required a human handoff."""

    model_config = ConfigDict(extra="forbid")

    trigger_turn: int = Field(
        description="The 1-based turn where the policy required escalation."
    )
    reason: str = Field(description="Why the policy required escalation.")
    escalation_turn: int | None = Field(
        description=(
            "The first 1-based turn at or after the trigger where the "
            "assistant handed off to a human, or null if it never did."
        ),
    )
    offered_turn: int | None = Field(
        description=(
            "The first 1-based turn where the assistant only offered or "
            "promised a handoff without making it, or null."
        ),
    )
    evidence: str = Field(
        description="Concise evidence from the transcript for this trigger."
    )


class UnnecessaryEscalation(BaseModel):
    """A handoff to a human that the escalation policy did not require."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(description="The 1-based turn of the handoff.")
    reason: str = Field(description="Why the handoff was not needed.")


class UnneededOffer(BaseModel):
    """An offer of a human that was not needed and never acted on."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(description="The 1-based turn of the offer.")
    reason: str = Field(description="Why no handoff was needed.")


class EscalationCorrectnessResponse(BaseModel):
    """Structured escalation evaluation output."""

    model_config = ConfigDict(extra="forbid")

    triggers: list[EscalationTrigger] = Field(
        description="Each turn where the policy required escalation, in order."
    )
    unnecessary_escalations: list[UnnecessaryEscalation] = Field(
        description="Each handoff the policy did not require, in order."
    )
    unneeded_offers: list[UnneededOffer] = Field(
        description=(
            "Each offer of a human, never acted on, made when the policy "
            "did not require a handoff, in order."
        )
    )
