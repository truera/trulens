"""Unit tests for the instruction retention conversation metric.

No external service is contacted: the provider is mocked at the transport
boundary and the judge answers with fixed JSON, so these run in the
``make test-unit`` sweep that CI executes.
"""

import json
from typing import ClassVar

import pytest
from trulens.feedback import llm_provider
from trulens.feedback.templates import conversation as templates_conversation

JSON_RULE = "Reply in JSON."
EU_RULE = "Only cover the EU."


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _JudgeProvider(llm_provider.LLMProvider):
    """Answers every judge call with the same reply and records the prompts."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, reply):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "messages", [])
        object.__setattr__(
            self,
            "_reply",
            reply if isinstance(reply, str) else json.dumps(reply),
        )

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        self.messages.append(messages)
        return self._reply

    def judge_prompt(self) -> str:
        return "\n".join(m["content"] for m in self.messages[-1])


def _verdicts(followed: dict[int, bool]) -> list[dict]:
    return [
        {"turn": turn, "followed": ok, "reason": "judged"}
        for turn, ok in followed.items()
    ]


def _judged(*instructions: tuple[str, int, dict[int, bool]]) -> dict:
    return {
        "instructions": [
            {"instruction": text, "turn_given": given, "verdicts": _verdicts(v)}
            for text, given, v in instructions
        ]
    }


CONVERSATION = [
    {"input": f"{JSON_RULE} List three EU capitals.", "output": '{"a": 1}'},
    {"input": "Add one more.", "output": '{"b": 2}'},
    {"input": "And one more.", "output": "Sure: Lisbon."},
]


def test_an_instruction_broken_in_turn_3_is_first_broken_at_turn_3():
    """The acceptance fixture: set in turn 1, kept in turn 2, broken in 3."""
    provider = _JudgeProvider(
        _judged((JSON_RULE, 1, {1: True, 2: True, 3: False}))
    )

    score, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert score == pytest.approx(2 / 3)
    [verdict] = meta["instructions"]
    assert verdict["instruction"] == JSON_RULE
    assert verdict["first_broken_turn"] == 3
    assert verdict["decided_by"] == "judge"
    assert meta["forgetting_ratio"] == pytest.approx(1 / 2)


def test_the_judge_sees_numbered_turns():
    """Verdicts are matched to turns by number, so the transcript numbers them."""
    provider = _JudgeProvider(_judged())
    messages = [
        {"role": "user", "content": JSON_RULE},
        {"role": "assistant", "content": "{}"},
        {"role": "user", "content": "Again."},
        {"role": "assistant", "content": "{}"},
    ]

    provider.instruction_retention_with_cot_reasons(messages)

    prompt = provider.judge_prompt()
    assert f"Turn 1 User: {JSON_RULE}" in prompt
    assert "Turn 2 Assistant: {}" in prompt


def test_a_check_decides_its_instruction_instead_of_the_judge():
    """A mechanical check is the source of truth, whatever the judge says."""

    def is_json(reply: str) -> bool:
        try:
            json.loads(reply)
        except ValueError:
            return False
        return True

    # The judge wrongly says turn 3 kept the rule; the check must win.
    provider = _JudgeProvider(
        _judged((JSON_RULE, 1, {1: True, 2: True, 3: True}))
    )

    score, meta = provider.instruction_retention_with_cot_reasons(
        CONVERSATION, checks={JSON_RULE: is_json}
    )

    [verdict] = meta["instructions"]
    assert verdict["decided_by"] == "check"
    assert verdict["first_broken_turn"] == 3
    assert score == pytest.approx(2 / 3)
    assert JSON_RULE in provider.judge_prompt()


def test_a_revoked_instruction_is_not_checked_after_its_revocation():
    """Turns after a revocation are not in force, so they cannot be broken."""
    provider = _JudgeProvider(
        _judged((EU_RULE, 1, {1: True, 2: True, 3: False}))
    )

    score, meta = provider.instruction_retention_with_cot_reasons(
        CONVERSATION, revocations={EU_RULE: 2}
    )

    assert score == 1.0
    [verdict] = meta["instructions"]
    assert verdict["revoked_turn"] == 2
    assert verdict["first_broken_turn"] is None
    assert [v["turn"] for v in verdict["verdicts"]] == [1, 2]


def test_a_dropped_instruction_that_comes_back_counts_as_corrected():
    """Followed, missed, followed: one forgetting and one correction."""
    provider = _JudgeProvider(
        _judged((EU_RULE, 1, {1: True, 2: False, 3: True}))
    )

    _, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert meta["forgetting_ratio"] == 1.0
    assert meta["correction_ratio"] == 1.0
    assert meta["instructions"][0]["first_broken_turn"] == 2


def test_verdicts_for_turns_before_the_instruction_are_ignored():
    """An instruction is in force from the turn it was given."""
    provider = _JudgeProvider(
        _judged(("Be brief.", 2, {1: False, 2: True, 3: True}))
    )

    score, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert score == 1.0
    assert [v["turn"] for v in meta["instructions"][0]["verdicts"]] == [2, 3]


def test_a_conversation_without_standing_instructions_scores_full_marks():
    """Nothing was asked, so nothing was forgotten."""
    provider = _JudgeProvider(_judged())

    score, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert score == 1.0
    assert meta["instructions"] == []


def test_an_unreadable_judge_answer_is_unparsable_not_a_score():
    """Prose instead of JSON must not be reported as a retention score."""
    provider = _JudgeProvider("The assistant mostly kept the rules.")

    score, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert score == llm_provider.UNPARSABLE_SCORE
    assert "error" in meta


def test_json_in_a_markdown_code_fence_is_read():
    """Models without structured outputs often fence their JSON."""
    answer = json.dumps(_judged((JSON_RULE, 1, {1: True, 2: False})))
    provider = _JudgeProvider(f"```json\n{answer}\n```")

    score, meta = provider.instruction_retention_with_cot_reasons(CONVERSATION)

    assert score == 0.5
    assert meta["instructions"][0]["first_broken_turn"] == 2


def test_checks_need_structured_records_not_a_transcript_string():
    """A check runs on each reply, which a flat transcript cannot give."""
    provider = _JudgeProvider(_judged())

    with pytest.raises(ValueError, match="checks"):
        provider.instruction_retention_with_cot_reasons(
            "Turn 1 User: hi", checks={JSON_RULE: lambda reply: True}
        )


def test_the_template_is_exported():
    assert "InstructionRetention" in templates_conversation.__all__
