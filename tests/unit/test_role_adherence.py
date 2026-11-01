"""Unit tests for the role adherence conversation metric."""

import json
from unittest import mock

import pytest
from trulens.feedback import llm_provider
from trulens.feedback import output_schemas
from trulens.feedback.templates import conversation as templates_conversation

_ROLE = (
    "A math tutor who guides with questions and never states the final answer."
)

_TUTOR_RECORDS = [
    {"input": "What is 12 x 12?", "output": "What is 12 x 10 plus 12 x 2?"},
    {"input": "120 + 24?", "output": "Good. What does that add up to?"},
    {"input": "Just tell me.", "output": "Fine, the answer is 144."},
]


class _ScriptedJudgeProvider(llm_provider.LLMProvider):
    """Provider whose judge replies with a fixed value and records its calls."""

    model_config = {"extra": "allow"}

    def __init__(self, reply):
        super().__init__(endpoint=None, model_engine="scripted-judge")
        endpoint = mock.MagicMock()
        endpoint.run_in_pace.side_effect = lambda func, *args, **kwargs: func(
            *args, **kwargs
        )
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "calls", [])

    def _create_chat_completion(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.reply


def _verdicts_json(*adheres: bool) -> str:
    return json.dumps({
        "turns": [
            {"turn": i, "adheres": ok, "reason": f"reason {i}"}
            for i, ok in enumerate(adheres, start=1)
        ]
    })


def test_tutor_fixture_scores_two_thirds_and_names_turn_three() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, True, False))

    score, meta = provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE
    )

    assert score == pytest.approx(2 / 3)
    assert meta["broken_turns"] == [3]
    assert [t["adheres"] for t in meta["turns"]] == [True, True, False]
    assert "Adhering turns: 2 of 3" in meta["reason"]
    assert "Turn 3 BREAKS: reason 3" in meta["reason"]


def test_one_judge_call_carries_role_transcript_and_schema() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, True, True))

    provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE, temperature=0.4
    )

    assert len(provider.calls) == 1
    call = provider.calls[0]
    system, user = (m["content"] for m in call["messages"])
    assert _ROLE in system
    assert "Turn 3 Assistant: Fine, the answer is 144." in user
    assert call["response_format"] is output_schemas.RoleAdherenceResponse
    assert call["temperature"] == 0.4


def test_additional_instructions_reach_the_judge() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, True, True))
    extra = "Hints that end in a question still count as guiding."

    provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE, additional_instructions=extra
    )

    assert extra in provider.calls[0]["messages"][0]["content"]


def test_every_turn_adhering_scores_one_and_every_turn_breaking_scores_zero() -> (
    None
):
    good = _ScriptedJudgeProvider(_verdicts_json(True, True, True))
    bad = _ScriptedJudgeProvider(_verdicts_json(False, False, False))

    assert good.role_adherence_with_cot_reasons(_TUTOR_RECORDS, _ROLE)[0] == 1.0
    score, meta = bad.role_adherence_with_cot_reasons(_TUTOR_RECORDS, _ROLE)
    assert score == 0.0
    assert meta["broken_turns"] == [1, 2, 3]


@pytest.mark.parametrize("role", ["", "   ", None])
def test_missing_role_raises_a_clear_error(role) -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, True, True))

    with pytest.raises(ValueError, match="`role` is required"):
        provider.role_adherence_with_cot_reasons(_TUTOR_RECORDS, role=role)

    assert provider.calls == []


def test_text_verdict_lines_are_read_when_the_judge_cannot_use_json() -> None:
    provider = _ScriptedJudgeProvider(
        "Turn 1: ADHERES - asks a guiding question\n"
        "Turn 2: **ADHERES** - nudges the student on\n"
        "Turn 3: BREAKS - states the final answer\n"
    )

    score, meta = provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE
    )

    assert score == pytest.approx(2 / 3)
    assert meta["broken_turns"] == [3]
    assert meta["turns"][2]["reason"] == "states the final answer"


def test_structured_response_object_is_accepted() -> None:
    reply = output_schemas.RoleAdherenceResponse(
        turns=[
            output_schemas.TurnVerdict(turn=1, adheres=True, reason="ok"),
            output_schemas.TurnVerdict(turn=2, adheres=True, reason="ok"),
            output_schemas.TurnVerdict(turn=3, adheres=False, reason="gave it"),
        ]
    )
    provider = _ScriptedJudgeProvider(reply)

    score, _ = provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE
    )

    assert score == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "reply",
    [
        "I think the assistant did fine.",
        _verdicts_json(True, True),  # one assistant turn missing
        _verdicts_json(True, True, True, True),  # a turn that does not exist
        json.dumps({
            "turns": [
                {"turn": 1, "adheres": True, "reason": "a"},
                {"turn": 1, "adheres": False, "reason": "b"},
                {"turn": 3, "adheres": True, "reason": "c"},
            ]
        }),  # turn 1 twice, turn 2 missing
    ],
    ids=["prose", "missing-turn", "extra-turn", "duplicate-turn"],
)
def test_unreadable_or_incomplete_verdicts_are_unparsable(reply) -> None:
    provider = _ScriptedJudgeProvider(reply)

    score, meta = provider.role_adherence_with_cot_reasons(
        _TUTOR_RECORDS, role=_ROLE
    )

    assert score == llm_provider.UNPARSABLE_SCORE
    assert "Could not read one verdict per assistant turn" in meta["reason"]


def test_transcript_string_uses_the_judges_turn_numbers() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, False))

    score, meta = provider.role_adherence_with_cot_reasons(
        "User: Hi\nAssistant: Hello\nUser: Joke?\nAssistant: Sure, a long one",
        role="A terse assistant.",
    )

    assert score == 0.5
    assert meta["broken_turns"] == [2]


def test_assistant_turn_numbers() -> None:
    assert templates_conversation.assistant_turn_numbers(_TUTOR_RECORDS) == [
        1,
        2,
        3,
    ]
    assert templates_conversation.assistant_turn_numbers([
        {"input": "a", "output": "b"},
        {"input": "c"},
        {"input": "d", "output": "e"},
    ]) == [1, 3]
    assert templates_conversation.assistant_turn_numbers("a transcript") is None


def test_assistant_turn_numbers_counts_role_content_messages() -> None:
    numbers = templates_conversation.assistant_turn_numbers
    assert numbers([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "joke?"},
        {"role": "Assistant", "content": "sure"},
    ]) == [1, 2]
    assert numbers([{"role": "user", "content": "hi"}]) == []
    assert numbers([{"speaker": "assistant", "text": "hello"}]) == [1]
    # Mixed styles and unknown shapes cannot be counted.
    assert (
        numbers([
            {"input": "a", "output": "b"},
            {"role": "assistant", "content": "hello"},
        ])
        is None
    )
    assert numbers(["not a record"]) is None


_MESSAGES = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "Tell me a secret."},
    {"role": "assistant", "content": "Sure, here is one."},
]


def test_role_content_messages_are_numbered_by_assistant_order() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, False))

    score, meta = provider.role_adherence_with_cot_reasons(
        _MESSAGES, role="A discreet concierge."
    )

    assert score == 0.5
    assert meta["broken_turns"] == [2]
    system = provider.calls[0]["messages"][0]["content"]
    assert "number the Assistant turns 1, 2, 3" in system


@pytest.mark.parametrize(
    "reply",
    [
        _verdicts_json(True),  # missing the second assistant message
        _verdicts_json(
            True, True, False
        ),  # a verdict for a third that is absent
    ],
    ids=["missing-verdict", "extra-verdict"],
)
def test_role_content_messages_need_one_verdict_per_assistant_message(
    reply,
) -> None:
    provider = _ScriptedJudgeProvider(reply)

    score, meta = provider.role_adherence_with_cot_reasons(
        _MESSAGES, role="A discreet concierge."
    )

    assert score == llm_provider.UNPARSABLE_SCORE
    assert "Could not read one verdict per assistant turn" in meta["reason"]


def test_a_single_assistant_message_rejects_an_extra_verdict() -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True, False))

    score, _ = provider.role_adherence_with_cot_reasons(
        [{"role": "assistant", "content": "hello"}], role="A greeter."
    )

    assert score == llm_provider.UNPARSABLE_SCORE


@pytest.mark.parametrize(
    "records",
    [
        [{"role": "user", "content": "Hi"}],
        [{"input": "Hi"}],
    ],
    ids=["role-content", "input-only-record"],
)
def test_no_assistant_turns_is_unparsable_without_a_judge_call(records) -> None:
    provider = _ScriptedJudgeProvider(_verdicts_json(True))

    score, meta = provider.role_adherence_with_cot_reasons(
        records, role="A greeter."
    )

    assert score == llm_provider.UNPARSABLE_SCORE
    assert "no assistant turns" in meta["reason"]
    assert provider.calls == []
