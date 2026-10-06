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
    "InstructionRetention",
    "TopicAdherence",
    "conversation_to_prompt",
    "conversation_turns",
    "turns_to_prompt",
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


def conversation_turns(
    records: list[Any],
) -> list[tuple[str | None, str | None]]:
    """Split conversation records into numbered (user, assistant) turns.

    Accepts the same record shapes as `conversation_to_prompt`. Role-based
    messages are paired: a user message opens a turn and the next assistant
    message closes it. System messages are not turns.

    Args:
        records: The ordered conversation records or messages.

    Returns:
        One `(user, assistant)` pair per turn; turn `n` is at index `n - 1`.
    """
    turns: list[tuple[str | None, str | None]] = []
    for record in records:
        if hasattr(record, "main_input") and hasattr(record, "main_output"):
            turns.append((
                _text(getattr(record, "main_input", None)),
                _text(getattr(record, "main_output", None)),
            ))
        elif isinstance(record, dict) and (
            "input" in record or "output" in record
        ):
            turns.append((
                _text(record.get("input")),
                _text(record.get("output")),
            ))
        elif isinstance(record, dict):
            role = str(record.get("role", record.get("speaker", ""))).lower()
            content = _text(
                record.get("content", record.get("text", record.get("message")))
            )
            if role == "system":
                continue
            if role == "assistant" and turns and turns[-1][1] is None:
                turns[-1] = (turns[-1][0], content)
            elif role == "assistant":
                turns.append((None, content))
            else:
                turns.append((content, None))
        else:
            turns.append((_text(record), None))
    return turns


def turns_to_prompt(turns: list[tuple[str | None, str | None]]) -> str:
    """Serialize `conversation_turns` output into a numbered transcript."""
    lines: list[str] = []
    for number, (user, assistant) in enumerate(turns, start=1):
        if user is not None:
            lines.append(f"Turn {number} User: {user}")
        if assistant is not None:
            lines.append(f"Turn {number} Assistant: {assistant}")
    return "\n".join(lines)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


class ConversationHelpfulness(Semantics, CriteriaOutputSpaceMixin):
    """Evaluates helpfulness across a multi-turn conversation."""

    criteria: ClassVar[str] = (
        "Is the assistant helpful, clear, and thorough across the conversation?"
    )
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

    criteria: ClassVar[str] = (
        "Does the conversation adhere to the specified reference topics?"
    )
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

    criteria: ClassVar[str] = (
        "Is the conversation logically coherent and consistent across turns?"
    )
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


class InstructionRetention(Semantics):
    """Evaluates whether standing instructions hold across conversation turns.

    The judge lists the standing instructions and rules on each assistant
    turn. Revocations are supplied by the caller and are not judged.
    """

    system_prompt: ClassVar[str] = cleandoc(
        """
        You are evaluating INSTRUCTION RETENTION in a multi-turn conversation
        between a User and an AI Assistant.

        A standing instruction is a constraint the user sets that should keep
        applying to the assistant's later replies: a required format ("answer
        in JSON"), a scope ("only cover the EU") or a prohibition ("never
        suggest X"). A one-off request that is answered and done is not a
        standing instruction.

        1. List every standing instruction the user gave, with the number of
           the turn it was given in.
        2. For each one, judge every assistant reply from that turn on: did
           the reply follow the instruction?

        Judge only whether each reply followed each instruction. Do not
        decide whether the user later cancelled an instruction: judge every
        reply from the turn it was given as if it still applies.

        Respond ONLY with JSON of this shape:
        {"instructions": [{"instruction": "...", "turn_given": 1,
          "verdicts": [{"turn": 1, "followed": true, "reason": "..."}]}]}
        With no standing instructions, respond {"instructions": []}.
        """
    )
    user_prompt_template: ClassVar[str] = cleandoc(
        """
        Conversation Transcript:
        {transcript}
        """
    )
    checked_instructions_template: ClassVar[str] = (
        "\n\nThese instructions are verified separately. Do not list them:\n"
        "{instructions}"
    )
