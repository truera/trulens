"""Tests for deterministic (non-LLM) feedback metrics."""

from trulens.feedback.deterministic import conversation_repetition
from trulens.feedback.deterministic import tool_argument_validity


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

    def test_score_within_unit_interval(self):
        records = _records([
            ("a", "b c d e f g h"),
            ("i", "j k l m n o p"),
        ])
        score, _ = conversation_repetition(records)
        assert 0.0 <= score <= 1.0


def test_tool_argument_validity_excludes_unknown_schemas():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    calls = [
        {"span_id": "one", "name": "search", "arguments": {"query": "a"}},
        {"span_id": "two", "name": "search", "arguments": {}},
        {
            "span_id": "three",
            "name": "search",
            "arguments": {"query": "c"},
        },
        {"span_id": "unknown", "name": "other", "arguments": {}},
    ]

    score, meta = tool_argument_validity(calls, {"search": schema})

    assert score == 0.67
    assert meta["num_calls_with_known_schema"] == 3
    assert meta["num_valid_calls"] == 2
    assert meta["invalid_calls"][0]["span_id"] == "two"
    assert "query" in meta["invalid_calls"][0]["errors"][0]
