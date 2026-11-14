"""Tests for deterministic (non-LLM) feedback metrics."""

import pytest
from trulens.feedback import deterministic as deterministic_feedback
from trulens.feedback.deterministic import conversation_repetition


def _records(pairs):
    """Build input/output-dict conversation records from (user, assistant)."""
    return [{"input": user, "output": assistant} for user, assistant in pairs]


class TestConversationRepetition:
    @pytest.mark.parametrize("reply", ["नमस्ते", "வணக்கம்", "مَرْحَبًا", "สวัสดี"])
    def test_identical_replies_with_combining_marks(self, reply):
        score, meta = conversation_repetition([
            {"output": reply},
            {"output": reply},
        ])
        assert score == 0.0
        assert meta["turns"][2]["overlap_with_previous"] == 1.0

    def test_distinct_vowel_signs_do_not_create_identical_replies(self):
        score, meta = conversation_repetition([
            {"output": "की"},
            {"output": "को"},
        ])
        assert score == 1.0
        assert meta["turns"][2]["overlap_with_previous"] == 0.0

    def test_combining_marks_do_not_create_low_diversity(self):
        reply = "உங்கள் சிக்கலை விவரிக்கவும். எங்கள் ஆதரவு குழு விரைவில் தொடர்பு கொள்ளும்."
        score, meta = conversation_repetition([{"output": reply}])
        assert score == 1.0
        assert meta["turns"][1]["lexical_diversity"] == 0.0
        assert meta["turns"][1]["ngram_repetition"] == 0.0

    @pytest.mark.parametrize(
        "reply",
        ["Hello world", "Привет мир", "你好世界", "Hello мир 你好", "界"],
        ids=["english", "cyrillic", "chinese", "mixed", "single-character"],
    )
    def test_identical_unicode_replies(self, reply):
        score, meta = conversation_repetition([
            {"output": reply},
            {"output": reply},
        ])
        assert score == 0.0
        assert meta["turns"][2]["overlap_with_previous"] == 1.0
        assert meta["turns"][2]["overlaps_turn"] == 1

    @pytest.mark.parametrize(
        "first, second",
        [("Привет мир", "Добрый вечер"), ("你好世界", "再见朋友")],
        ids=["cyrillic", "chinese"],
    )
    def test_distinct_unicode_replies(self, first, second):
        score, meta = conversation_repetition([
            {"output": first},
            {"output": second},
        ])
        assert score == 1.0
        assert meta["turns"][2]["overlap_with_previous"] == 0.0

    def test_distinct_unicode_after_shared_ascii_prefix(self):
        score, meta = conversation_repetition([
            {"output": "Answer: Привет"},
            {"output": "Answer: Пока"},
        ])
        assert score == 0.75
        assert meta["turns"][2]["overlap_with_previous"] == 0.25

    @pytest.mark.parametrize("reply", ["", " \t", "...___!", "Привет мир"])
    def test_single_reply_has_no_repetition(self, reply):
        score, meta = conversation_repetition([{"output": reply}])
        assert score == (None if not reply else 1.0)
        assert meta["num_assistant_turns"] == (0 if not reply else 1)

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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("HELLO_world42, 123!", ["hello", "world42", "123"]),
        ("ПРИВЕТ_мир!", ["привет", "мир"]),
        ("你好，世界！", ["你好", "世界"]),
        ("Hello_мир_你好42", ["hello", "мир", "你好42"]),
        ("123_٤٥٦_１２３_²", ["123", "٤٥٦", "１２３", "²"]),
        ("", []),
        (" \t\n", []),
        ("...___!?", []),
    ],
)
def test_tokenize_alphanumeric_text(text, expected):
    assert deterministic_feedback._tokenize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("नमस्ते", ["नमस्ते"]),
        ("வணக்கம்", ["வணக்கம்"]),
        ("مَرْحَبًا", ["مَرْحَبًا"]),
        ("สวัสดี", ["สวัสดี"]),
        ("CAFE\u0301", ["cafe\u0301"]),
        ("1\u20dd", ["1\u20dd"]),
        ("\u0301\u093e\u20dd", []),
        ("\u0301a\u0301_\u0301b!\u0301c \u0301d", ["a\u0301", "b", "c", "d"]),
    ],
)
def test_tokenize_combining_marks(text, expected):
    assert deterministic_feedback._tokenize(text) == expected
