"""Unit tests for conversation-aware metric templates."""

from unittest import mock

from trulens.feedback import llm_provider
from trulens.feedback.templates import conversation as templates_conversation


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


def test_agent_goal_accuracy_requests_score_only() -> None:
    prompt = templates_conversation.AgentGoalAccuracy.system_prompt_template
    assert "Reason:" not in prompt
    assert "Respond ONLY" in prompt
