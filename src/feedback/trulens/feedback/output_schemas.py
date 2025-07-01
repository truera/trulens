from pydantic import BaseModel
from pydantic import Field


class BaseFeedbackResponse(BaseModel):
    """
    A base model for feedback responses.
    It can be extended to include specific fields for different feedback types.
    """

    score: int = Field(description="The score based on the given criteria.")


class ChainOfThoughtResponse(BaseModel):
    """
    A model to represent the response from a Chain of Thought (COT) evaluation.
    It includes the criteria, supporting evidence, and score.
    """

    criteria: str = Field(description="The criteria for the evaluation.")
    supporting_evidence: str = Field(
        description="Supporting evidence for the score, detailing the reasoning step by step."
    )
    score: int = Field(description="The score based on the given criteria.")
