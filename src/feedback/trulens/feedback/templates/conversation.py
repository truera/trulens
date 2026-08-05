"""Conversation-aware evaluation templates."""

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
    "TopicAdherence",
    "conversation_to_prompt",
]


def conversation_to_prompt(records: list[Any] | str) -> str:
    """Serialize conversation records or messages into a transcript."""
    if isinstance(records, str):
        return records

    transcript_lines: list[str] = []
    for idx, record in enumerate(records, start=1):
        if hasattr(record, "main_input") and hasattr(record, "main_output"):
            user_input = getattr(record, "main_input", None)
            assistant_output = getattr(record, "main_output", None)
            if user_input is not None:
                transcript_lines.append(f"Turn {idx} User: {user_input}")
            if assistant_output is not None:
                transcript_lines.append(
                    f"Turn {idx} Assistant: {assistant_output}"
                )
        elif isinstance(record, dict) and (
            "input" in record or "output" in record
        ):
            if record.get("input") is not None:
                transcript_lines.append(f"Turn {idx} User: {record['input']}")
            if record.get("output") is not None:
                transcript_lines.append(
                    f"Turn {idx} Assistant: {record['output']}"
                )
        elif isinstance(record, dict):
            role = record.get("role", record.get("speaker", f"Turn {idx}"))
            content = record.get(
                "content", record.get("text", record.get("message", ""))
            )
            transcript_lines.append(f"{str(role).capitalize()}: {content}")
        else:
            transcript_lines.append(f"Turn {idx}: {record!s}")

    return "\n".join(transcript_lines)


class ConversationHelpfulness(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates helpfulness across a multi-turn conversation."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    system_prompt: ClassVar[str] = cleandoc(
        f"""
        You are evaluating the overall HELPFULNESS of a multi-turn conversation
        between a User and an AI Assistant. Score the assistant's helpfulness
        across all turns on a scale from 0 to 3:
        0: Not helpful at all or misleading.
        1: Partially helpful but missed key user needs.
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


class TopicAdherence(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates topic adherence across conversation turns."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    system_prompt_template: ClassVar[str] = cleandoc(
        f"""
        You are evaluating TOPIC ADHERENCE in a multi-turn conversation.
        Determine how closely the conversation adheres to these topics:
        {{reference_topics}}.
        0: Completely off-topic.
        1: Barely touches the topics with major deviations.
        2: Mostly adheres with minor tangents.
        3: Strongly adheres throughout the conversation.

        Respond ONLY with a single integer score from {LIKERT_0_3_PROMPT}.
        """
    )


class AgentGoalAccuracy(Semantics):
    """Evaluates binary goal completion for an agent conversation."""

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        You are evaluating whether the AI Assistant successfully fulfilled the
        user's goal in the conversation.
        Goal / Reference: {reference_goal}

        Respond ONLY with 1 if the goal was achieved, or 0 if it failed or
        remained incomplete.
        """
    )


class CoherenceAcrossTurns(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates logical coherence across conversation turns."""

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    system_prompt: ClassVar[str] = cleandoc(
        f"""
        You are evaluating COHERENCE ACROSS TURNS in a multi-turn conversation.
        0: Severe contradictions or incoherent jumps.
        1: Frequent loss of context or minor contradictions.
        2: Mostly coherent with minor conversational friction.
        3: Flawless flow, context retention, and logical consistency.

        Respond ONLY with a single integer score from {LIKERT_0_3_PROMPT}.
        """
    )
