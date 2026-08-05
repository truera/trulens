"""Unit tests for conversation-aware metric templates."""

from unittest.mock import MagicMock

import pytest
from trulens.feedback.llm_provider import LLMProvider
from trulens.feedback.templates.conversation import AgentGoalAccuracy
from trulens.feedback.templates.conversation import CoherenceAcrossTurns
from trulens.feedback.templates.conversation import ConversationHelpfulness
from trulens.feedback.templates.conversation import TopicAdherence
from trulens.feedback.templates.conversation import conversation_to_prompt


class TestConversationTemplates:
    def test_conversation_to_prompt_string(self):
        transcript = "User: Hello\nAssistant: Hi there!"
        result = conversation_to_prompt(transcript)
        assert result == transcript

    def test_conversation_to_prompt_dicts(self):
        records = [
            {"role": "user", "content": "How's the weather?"},
            {"role": "assistant", "content": "It is sunny today."},
        ]
        result = conversation_to_prompt(records)
        assert "User: How's the weather?" in result
        assert "Assistant: It is sunny today." in result

    def test_conversation_to_prompt_record_objects(self):
        class MockRecord:
            def __init__(self, inp, out):
                self.main_input = inp
                self.main_output = out

        records = [
            MockRecord("What is 2+2?", "2+2 equals 4."),
            MockRecord("Thanks!", "You are welcome!"),
        ]
        result = conversation_to_prompt(records)
        assert "Turn 1 User: What is 2+2?" in result
        assert "Turn 1 Assistant: 2+2 equals 4." in result

    def test_conversation_to_prompt_fallback(self):
        records = ["Simple text turn"]
        result = conversation_to_prompt(records)
        assert "Turn 1: Simple text turn" in result

    def test_provider_conversation_helpfulness(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 0.95

        records = [{"role": "user", "content": "Help me code."}]
        score = LLMProvider.conversation_helpfulness(
            mock_provider, records=records, custom_arg="value"
        )
        assert score == 0.95
        mock_provider.generate_score.assert_called_once()
        _, kwargs = mock_provider.generate_score.call_args
        assert kwargs["min_score_val"] == 0
        assert kwargs["max_score_val"] == 3
        assert kwargs["custom_arg"] == "value"

    def test_provider_topic_adherence(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 0.85

        records = [{"role": "user", "content": "Let's discuss banking."}]
        score = LLMProvider.topic_adherence(
            mock_provider,
            records=records,
            reference_topics=["banking", "finance"],
            custom_arg="value",
        )
        assert score == 0.85
        mock_provider.generate_score.assert_called_once()
        _, kwargs = mock_provider.generate_score.call_args
        assert kwargs["min_score_val"] == 0
        assert kwargs["max_score_val"] == 3
        assert kwargs["custom_arg"] == "value"

    def test_provider_agent_goal_accuracy(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 1.0

        records = [{"role": "user", "content": "Book a flight."}]
        score = LLMProvider.agent_goal_accuracy(
            mock_provider,
            records=records,
            reference_goal="Book flight",
            custom_arg="value",
        )
        assert score == 1.0
        mock_provider.generate_score.assert_called_once()
        _, kwargs = mock_provider.generate_score.call_args
        assert kwargs["min_score_val"] == 0
        assert kwargs["max_score_val"] == 1
        assert kwargs["custom_arg"] == "value"

    def test_provider_coherence_across_turns(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 0.9

        records = [{"role": "user", "content": "Tell me a joke."}]
        score = LLMProvider.coherence_across_turns(
            mock_provider, records=records, custom_arg="value"
        )
        assert score == 0.9
        mock_provider.generate_score.assert_called_once()
        _, kwargs = mock_provider.generate_score.call_args
        assert kwargs["min_score_val"] == 0
        assert kwargs["max_score_val"] == 3
        assert kwargs["custom_arg"] == "value"

    def test_provider_error_propagation(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.side_effect = RuntimeError("LLM Failure")

        records = [{"role": "user", "content": "Test error"}]
        with pytest.raises(RuntimeError, match="LLM Failure"):
            LLMProvider.conversation_helpfulness(mock_provider, records=records)

        with pytest.raises(RuntimeError, match="LLM Failure"):
            LLMProvider.topic_adherence(
                mock_provider,
                records=records,
                reference_topics=["test"],
            )

    def test_template_structures(self):
        assert hasattr(ConversationHelpfulness, "system_prompt")
        assert hasattr(TopicAdherence, "system_prompt_template")
        assert hasattr(AgentGoalAccuracy, "system_prompt_template")
        assert hasattr(CoherenceAcrossTurns, "system_prompt")
