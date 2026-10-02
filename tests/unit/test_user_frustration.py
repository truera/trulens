"""Unit tests for the user frustration conversation metric."""

from unittest import mock

from trulens.feedback import llm_provider
from trulens.feedback.templates import conversation as templates_conversation

_RECORDS = [
    {"input": "What is my account balance?", "output": "You have no balance."},
    {"input": "That is wrong, check again.", "output": "It still shows zero."},
    {"input": "What is my account balance?", "output": "Zero."},
]

_SATISFIED_RECORDS = [
    {"input": "Thanks, that solved it!", "output": "Glad I could help."},
]


def test_user_frustration_template_is_registered() -> None:
    assert "UserFrustration" in templates_conversation.__all__
    template = templates_conversation.UserFrustration
    # 0-3 Likert, normalized to a 0.0-1.0 score by the provider.
    assert template.output_space == "LIKERT_0_3"
    assert "0 to 3" in template.output_space_prompt
    # Judge is driven by the user's turns and recognizes each signal.
    system_prompt = template.system_prompt
    for signal in (
        "Repeated requests",
        "Corrections",
        "dissatisfaction",
        "Abandonment",
    ):
        assert signal in system_prompt


def test_user_frustration_uses_score_arguments() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 0.0

    # A user who corrects and repeats a request is scored below 0.5.
    assert (
        llm_provider.LLMProvider.user_frustration(provider, _RECORDS) == 0.0
    )
    provider.generate_score.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    )


def test_user_frustration_cot_returns_reasons() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score_and_reasons.return_value = (
        0.0,
        {"reason": "repeated request at turn 3"},
    )

    score, reasons = llm_provider.LLMProvider.user_frustration_with_cot_reasons(
        provider, _RECORDS
    )
    assert score == 0.0
    assert reasons == {"reason": "repeated request at turn 3"}
    provider.generate_score_and_reasons.assert_called_once_with(
        system_prompt=mock.ANY,
        user_prompt=mock.ANY,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    )


def test_user_frustration_satisfied_conversation_scores_high() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 1.0

    # A conversation ending in thanks with no corrections scores 1.0.
    assert (
        llm_provider.LLMProvider.user_frustration(
            provider, _SATISFIED_RECORDS
        )
        == 1.0
    )


def test_user_frustration_transcript_passes_user_turns() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 1.0

    llm_provider.LLMProvider.user_frustration(provider, _RECORDS)

    kwargs = provider.generate_score.call_args.kwargs
    # Both the corrected and repeated user turns are present in the prompt.
    assert "That is wrong, check again." in kwargs["user_prompt"]
    # user_prompt is built from the shared user prompt template.
    assert kwargs["user_prompt"].startswith("Conversation Transcript:")


def test_user_frustration_forwards_positional_temperature() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score_and_reasons.return_value = (1.0, {"reason": "ok"})

    llm_provider.LLMProvider.user_frustration_with_cot_reasons(
        provider, _RECORDS, 0.7
    )

    kwargs = provider.generate_score_and_reasons.call_args.kwargs
    assert kwargs["temperature"] == 0.7


def test_user_frustration_keeps_keyword_additional_instructions() -> None:
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.generate_score.return_value = 1.0
    extra = "Treat sarcasm as frustration."

    llm_provider.LLMProvider.user_frustration(
        provider, _RECORDS, additional_instructions=extra
    )

    assert (
        provider._build_criteria_with_instructions.call_args.kwargs[
            "additional_instructions"
        ]
        == extra
    )
