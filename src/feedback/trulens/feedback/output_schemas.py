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


class InstructionTurnVerdict(BaseModel):
    """Whether one assistant turn followed one standing instruction."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(description="The turn number of the assistant reply.")
    followed: bool = Field(
        description="Whether the reply in this turn followed the instruction."
    )
    reason: str = Field(description="One sentence on why.")


class StandingInstruction(BaseModel):
    """A standing instruction from the user and the verdict for each turn."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        description="The instruction, as the user gave it."
    )
    turn_given: int = Field(description="The turn the user gave it in.")
    verdicts: list[InstructionTurnVerdict] = Field(
        description="One verdict for each assistant turn from turn_given on."
    )


class InstructionRetentionResponse(BaseModel):
    """The standing instructions in a conversation, judged turn by turn."""

    model_config = ConfigDict(extra="forbid")

    instructions: list[StandingInstruction] = Field(
        description="Every standing instruction the user gave."
    )
