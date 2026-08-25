"""Unit tests for conversation-aware metric templates."""

from unittest import mock

import pytest
from trulens.feedback import llm_provider
from trulens.feedback.templates import conversation as templates_conversation

_RECORDS = [{"input": "Help", "output": "Done"}]

# Pre-#2710, temperature was the next positional after records (and any domain
# parameter). Inserting additional_instructions before it bound a positional
# temperature to additional_instructions and raised TypeError.
_POSITIONAL_TEMPERATURE_CASES = [
    (
        "conversation_helpfulness",
        (_RECORDS, 0.7),
        "generate_score",
        0,
        3,
    ),
    (
        "conversation_helpfulness_with_cot_reasons",
        (_RECORDS, 0.7),
        "generate_score_and_reasons",
        0,
        3,
    ),
    (
        "coherence_across_turns",
        (_RECORDS, 0.7),
        "generate_score",
        0,
        3,
    ),
    (
        "coherence_across_turns_with_cot_reasons",
        (_RECORDS, 0.7),
        "generate_score_and_reasons",
        0,
        3,
    ),
    (
        "topic_adherence",
        (_RECORDS, ["support"], 0.7),
        "generate_score",
        0,
        3,
    ),
    (
        "topic_adherence_with_cot_reasons",
        (_RECORDS, ["support"], 0.7),
        "generate_score_and_reasons",
        0,
        3,
    ),
    (
        "agent_goal_accuracy",
        (_RECORDS, "resolve the issue", 0.7),
        "generate_score",
        0,
        1,
    ),
    (
        "agent_goal_accuracy_with_cot_reasons",
        (_RECORDS, "resolve the issue", 0.7),
        "generate_score_and_reasons",
        0,
        1,
    ),
]


def test_conversation_to_prompt_records() -> None:
    records = [
        {"input": "What is 2+2?", "output": "Four."},
        {"input": "Why?", "output": "Two pairs total four items."},
    ]

    assert templates_conversation.conversation_to_prompt(records) == (
        "Turn 1 User: What is 2+2?\n"
        "Turn 1 Assistant: Four.\n"
        "Turn 2 User: Why?\n"
        "Turn 2 Assistant: Two pairs total four items."
    )


def test_conversation_provider_methods_use_supported_score_arguments() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 1.0
    records = [{"input": "Help", "output": "Done"}]

    assert (
        llm_provider.LLMProvider.topic_adherence(
            provider,
            records=records,
            reference_topics=["support"],
        )
        == 1.0
    )
    provider.generate_score.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    )


def test_conversation_cot_methods_use_supported_score_arguments() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score_and_reasons.return_value = (1.0, {"reason": "ok"})
    records = [{"input": "Help", "output": "Done"}]

    assert llm_provider.LLMProvider.topic_adherence_with_cot_reasons(
        provider,
        records=records,
        reference_topics=["support"],
    ) == (1.0, {"reason": "ok"})
    provider.generate_score_and_reasons.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    )


def test_agent_goal_accuracy_cot_keeps_binary_output_space() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score_and_reasons.return_value = (1.0, {"reason": "ok"})
    records = [{"input": "Help", "output": "Done"}]

    assert llm_provider.LLMProvider.agent_goal_accuracy_with_cot_reasons(
        provider,
        records=records,
    ) == (1.0, {"reason": "ok"})
    provider.generate_score_and_reasons.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=1,
        temperature=0.0,
    )


def test_agent_goal_accuracy_requests_score_only() -> None:
    prompt = templates_conversation.AgentGoalAccuracy.system_prompt_template
    assert "Reason:" not in prompt
    assert "Respond ONLY" in prompt


@pytest.mark.parametrize(
    "method_name, args, score_attr, min_score_val, max_score_val",
    _POSITIONAL_TEMPERATURE_CASES,
    ids=[case[0] for case in _POSITIONAL_TEMPERATURE_CASES],
)
def test_positional_temperature_is_forwarded_for_conversation_metrics(
    method_name: str,
    args: tuple,
    score_attr: str,
    min_score_val: int,
    max_score_val: int,
) -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    if score_attr == "generate_score":
        provider.generate_score.return_value = 1.0
    else:
        provider.generate_score_and_reasons.return_value = (
            1.0,
            {"reason": "ok"},
        )

    getattr(llm_provider.LLMProvider, method_name)(provider, *args)

    getattr(provider, score_attr).assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=min_score_val,
        max_score_val=max_score_val,
        temperature=0.7,
    )


def test_positional_temperature_keeps_keyword_additional_instructions() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 1.0
    extra = "Treat structured rows as coherent."

    llm_provider.LLMProvider.conversation_helpfulness(
        provider, _RECORDS, 0.7, additional_instructions=extra
    )

    provider.generate_score.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=3,
        temperature=0.7,
    )
    assert (
        provider._build_criteria_with_instructions.call_args.kwargs[
            "additional_instructions"
        ]
        == extra
    )
