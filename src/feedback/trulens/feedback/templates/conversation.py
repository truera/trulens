"""
Conversation-aware evaluation templates for multi-turn chat sessions.
Includes transcript renderer and multi-turn feedback templates:
- ConversationHelpfulness
- SimpleCriteriaScore
- TopicAdherence
- AgentGoalAccuracy
- CoherenceAcrossTurns
"""

from inspect import cleandoc
from typing import Any, ClassVar

from trulens.feedback.templates.base import LIKERT_0_3_PROMPT
from trulens.feedback.templates.base import CriteriaOutputSpaceMixin
from trulens.feedback.templates.base import OutputSpace
from trulens.feedback.templates.base import Semantics

__all__ = [
    "AgentGoalAccuracy",
    "CoherenceAcrossTurns",
    "ConversationHelpfulness",
    "SimpleCriteriaScore",
    "TopicAdherence",
    "conversation_to_prompt",
]


def conversation_to_prompt(records: list[Any] | str) -> str:
    """Serialize a list of Record objects or turn dictionaries into a human-readable transcript.

    Args:
        records: A list of Record objects, turn dicts, or a pre-formatted string.

    Returns:
        Formatted transcript string with role annotations (User, Assistant, Tool).
    """
    if isinstance(records, str):
        return records

    transcript_lines: list[str] = []

    for idx, rec in enumerate(records, start=1):
        if hasattr(rec, "main_input") and hasattr(rec, "main_output"):
            user_input = getattr(rec, "main_input", None)
            asst_output = getattr(rec, "main_output", None)
            if user_input:
                transcript_lines.append(f"Turn {idx} User: {user_input}")
            if asst_output:
                transcript_lines.append(f"Turn {idx} Assistant: {asst_output}")
        elif isinstance(rec, dict):
            role = rec.get("role", rec.get("speaker", f"Turn {idx}"))
            content = rec.get(
                "content", rec.get("text", rec.get("message", ""))
            )
            transcript_lines.append(f"{role.capitalize()}: {content}")
        else:
            transcript_lines.append(f"Turn {idx}: {rec!s}")

    return "\n".join(transcript_lines)


class ConversationHelpfulness(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates helpfulness across an entire multi-turn conversation."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    system_prompt: ClassVar[str] = cleandoc(
        f"""
        You are evaluating the overall HELPFULNESS of a multi-turn conversation between a User and an AI Assistant.
        Score the assistant's helpfulness across all turns on a scale from 0 to 3:
        0: Not helpful at all or misleading.
        1: Partially helpful but missed key user needs or left questions unanswered.
        2: Mostly helpful and resolved core questions with minor gaps.
        3: Extremely helpful, thorough, clear, and proactive.

        Respond ONLY with a single integer score from {LIKERT_0_3_PROMPT}.
        """
    )

    user_prompt_template: ClassVar[str] = cleandoc(
        """
        Conversation Transcript:
        {transcript}
        """
    )


class SimpleCriteriaScore(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates a multi-turn conversation against custom user-provided criteria."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    system_prompt_template: ClassVar[str] = cleandoc(
        f"""
        You are evaluating a multi-turn conversation against specific criteria.
        Respond ONLY with a number from {LIKERT_0_3_PROMPT} where 0 means does not satisfy the criteria at all and 3 fully satisfies the criteria.

        Criteria:
        {{criteria}}
        """
    )


class TopicAdherence(Semantics):
    """Evaluates topic adherence (precision, recall, f1) across conversation turns."""

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        You are evaluating TOPIC ADHERENCE across a multi-turn conversation against reference topics.
        Determine whether the conversation discussed the reference topics: {reference_topics}.

        Return JSON format:
        {{
          "precision": <float in 0-1>,
          "recall": <float in 0-1>,
          "f1": <float in 0-1>,
          "reason": "<explanation>"
        }}
        """
    )


class AgentGoalAccuracy(Semantics):
    """Evaluates binary goal completion (0 or 1) for agentic conversations."""

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        You are evaluating whether the AI Assistant successfully fulfilled the user's goal in the conversation.
        Goal / Reference: {reference_goal}

        Respond with 1 if the goal was successfully achieved, or 0 if it failed or remained incomplete.
        Format: SCORE: <0 or 1>
        Reason: <explanation>
        """
    )


class CoherenceAcrossTurns(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates consistency and logical coherence across conversation turns."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    system_prompt: ClassVar[str] = cleandoc(
        f"""
        You are evaluating COHERENCE ACROSS TURNS in a multi-turn conversation.
        Score how logically connected, non-contradictory, and smooth the dialogue is across turns:
        0: Severe contradictions or incoherent jumps between turns.
        1: Frequent loss of context or minor contradictions.
        2: Mostly coherent with minor conversational friction.
        3: Flawless topic flow, perfect context retention, and complete logical consistency.

        Respond ONLY with a single integer score from {LIKERT_0_3_PROMPT}.
        """
    )
