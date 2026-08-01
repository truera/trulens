"""Unit tests for conversation-aware metric templates."""

from unittest.mock import MagicMock

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

    def test_provider_conversation_helpfulness(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 0.95

        from trulens.feedback.llm_provider import LLMProvider

        # Call method directly bound or via mock
        records = [{"role": "user", "content": "Help me code."}]
        score = LLMProvider.conversation_helpfulness(
            mock_provider, records=records
        )
        assert score == 0.95
        assert mock_provider.generate_score.called

    def test_provider_topic_adherence(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 0.85

        from trulens.feedback.llm_provider import LLMProvider

        records = [{"role": "user", "content": "Let's discuss banking."}]
        res = LLMProvider.topic_adherence(
            mock_provider, records=records, reference_topics=["banking"]
        )
        assert res == {"precision": 0.85, "recall": 0.85, "f1": 0.85}

    def test_provider_agent_goal_accuracy(self):
        mock_provider = MagicMock()
        mock_provider.generate_score.return_value = 1.0

        from trulens.feedback.llm_provider import LLMProvider

        records = [{"role": "user", "content": "Book a flight."}]
        score = LLMProvider.agent_goal_accuracy(
            mock_provider, records=records, reference_goal="Book flight"
        )
        assert score == 1.0
