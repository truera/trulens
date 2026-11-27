"""Tests for deterministic (non-LLM) feedback metrics."""

from trulens.feedback.deterministic import conversation_repetition
from trulens.feedback.templates import conversation as templates_conversation


def _records(pairs):
    """Build input/output-dict conversation records from (user, assistant)."""
    return [{"input": user, "output": assistant} for user, assistant in pairs]


class TestConversationRepetition:
    def test_identical_replies_score_below_half(self):
        records = _records([
            (
                "What is the capital of France?",
                "The capital of France is Paris.",
            ),
            ("Tell me again.", "The capital of France is Paris."),
        ])
        score, meta = conversation_repetition(records)
        assert score < 0.5
        # Both turns are named in the per-turn breakdown.
        assert set(meta["turns"].keys()) == {1, 2}
        assert meta["turns"][2]["overlaps_turn"] == 1
        assert meta["worst_turn"] == 2

    def test_varied_conversation_scores_above_point_nine(self):
        records = _records([
            (
                "What is the capital of France?",
                "The capital of France is Paris, a city on the Seine.",
            ),
            (
                "What is its population?",
                "Roughly two million people live within the city limits.",
            ),
            (
                "And the country?",
                "France is in Western Europe and uses the euro.",
            ),
        ])
        score, meta = conversation_repetition(records)
        assert score > 0.9

    def test_fallback_phrase_reply_scores_low(self):
        records = _records([
            ("Hi there!", "Hello! How can I help you today?"),
            (
                "Tell me a secret.",
                "I'm sorry, I can't help with that.",
            ),
        ])
        score, meta = conversation_repetition(records)
        assert score < 0.5
        assert meta["turns"][2]["fallback_phrase"] == 1.0

    def test_role_content_dicts(self):
        records = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hi again"},
            {"role": "assistant", "content": "Hello!"},
        ]
        score, _ = conversation_repetition(records)
        assert score < 0.5

    def test_transcript_string(self):
        transcript = (
            "Turn 1 User: Hi\n"
            "Turn 1 Assistant: Hello!\n"
            "Turn 2 User: Hi again\n"
            "Turn 2 Assistant: Hello!"
        )
        score, meta = conversation_repetition(transcript)
        assert score < 0.5
        assert meta["num_assistant_turns"] == 2

    def test_transcript_scores_like_its_records(self):
        # A transcript used to keep only the first line of each reply, so
        # these two different replies looked like an exact repeat. A reply's
        # later lines, `Note:` included, belong to it, while the second line
        # of a user message and a message from any other speaker do not.
        replies = [
            "Here is what I suggest:\n"
            "a light raincoat and comfortable walking shoes\n"
            "Note: it rains often in spring",
            "Here is what I suggest:\n"
            "sunscreen, a hat and sandals for the warm afternoons",
        ]
        conversations = [
            _records([
                ("What should I pack for Paris?", replies[0]),
                ("And for Rome?\nWe go in July.", replies[1]),
            ]),
            [
                {"role": "user", "content": "What should I pack?"},
                {"role": "assistant", "content": replies[0]},
                {"role": "critic", "content": "Both lists are short."},
                {"role": "assistant", "content": replies[1]},
            ],
            [
                {"role": "assistant", "content": "same reply"},
                {"role": "critic", "content": "unique critique"},
                {"role": "assistant", "content": "same reply"},
            ],
        ]
        scores = []
        for records in conversations:
            transcript = templates_conversation.conversation_to_prompt(records)
            expected = conversation_repetition(records)
            assert conversation_repetition(transcript) == expected
            scores.append(expected[0])
        assert scores[0] > 0.5
        assert scores[1] > 0.5
        assert scores[2] == 0.0

    def test_no_assistant_turns_returns_none(self):
        score, meta = conversation_repetition([{"input": "hi"}])
        assert score is None
        assert meta["num_assistant_turns"] == 0

    def test_empty_conversation_returns_none(self):
        score, meta = conversation_repetition([])
        assert score is None
        assert meta["num_assistant_turns"] == 0

    def test_transcript_without_assistant_turns_returns_none(self):
        transcript = "USER: hi\nUSER: hello"
        score, meta = conversation_repetition(transcript)
        assert score is None
        assert meta["num_assistant_turns"] == 0

    def test_custom_fallback_phrases(self):
        records = _records([
            ("Hi", "Hello! How can I help you today?"),
            (
                "Are you there?",
                "Greetings, traveler. I can help with many things.",
            ),
        ])
        default_score, _ = conversation_repetition(records)
        custom_score, custom_meta = conversation_repetition(
            records, fallback_phrases=["greetings, traveler"]
        )
        assert custom_score < default_score
        assert custom_score < 0.5
        assert custom_meta["turns"][2]["fallback_phrase"] == 1.0

    def test_empty_fallback_phrases_disables_the_signal(self):
        records = _records([("Hi", "I don't understand, could you rephrase?")])
        _, meta = conversation_repetition(records, fallback_phrases=[])
        assert meta["turns"][1]["fallback_phrase"] == 0.0

    def test_score_within_unit_interval(self):
        records = _records([
            ("a", "b c d e f g h"),
            ("i", "j k l m n o p"),
        ])
        score, _ = conversation_repetition(records)
        assert 0.0 <= score <= 1.0
