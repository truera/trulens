"""Unit tests for memory recall evaluation via recall_at_k/precision_at_k/mrr with conversation_id.

These tests verify that the existing GroundTruthAgreement metrics (recall_at_k,
precision_at_k, mrr) correctly handle conversation-scoped memory recall
evaluation when conversation_id is provided.

The memory recall use case stores expected memory texts as expected_chunks
in the golden set, using conversation_id to scope lookups per conversation.
"""

import numpy as np
from trulens.feedback.groundtruth import GroundTruthAgreement

# ---------------------------------------------------------------------------
# Shared golden set fixture
# ---------------------------------------------------------------------------

MEMORY_GOLDEN_SET = [
    # Conversation 1: user preferences
    {
        "query": "What are the user's preferences?",
        "expected_chunks": [
            {"text": "User prefers dark mode", "expect_score": 1},
            {"text": "User likes Python", "expect_score": 1},
        ],
        "conversation_id": "conv_1",
    },
    {
        "query": "What programming languages does the user like?",
        "expected_chunks": [
            {"text": "User likes Python", "expect_score": 1},
        ],
        "conversation_id": "conv_1",
    },
    # Conversation 2: same query, different expected memories
    {
        "query": "What are the user's preferences?",
        "expected_chunks": [
            {"text": "User prefers light mode", "expect_score": 1},
            {"text": "User likes JavaScript", "expect_score": 1},
        ],
        "conversation_id": "conv_2",
    },
    # Conversation without conversation_id (legacy)
    {
        "query": "What happened yesterday?",
        "expected_chunks": [
            {"text": "Team standup at 9am", "expect_score": 1},
            {"text": "Deployed v2.1 to staging", "expect_score": 1},
        ],
    },
]


def _make_gta(conversation_id=None):
    """Create a GroundTruthAgreement with a dummy provider for testing."""
    from trulens.feedback.dummy.provider import DummyProvider

    provider = DummyProvider()
    return GroundTruthAgreement(
        MEMORY_GOLDEN_SET,
        provider=provider,
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# recall_at_k with conversation_id
# ---------------------------------------------------------------------------


class TestRecallAtKWithConversationId:
    def test_recall_at_k_scoped_to_conversation(self):
        """Same query in different conversations should return different results."""
        gta = _make_gta()

        # conv_1: expects dark mode + Python
        result_1 = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode", "User likes Python"],
            conversation_id="conv_1",
        )
        assert result_1 == 1.0

        # conv_2: expects light mode + JavaScript
        result_2 = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode", "User likes Python"],
            conversation_id="conv_2",
        )
        assert result_2 == 0.0  # wrong conversation's expected chunks

    def test_recall_at_k_instance_default(self):
        """Instance-level conversation_id should be used as default."""
        gta = _make_gta(conversation_id="conv_1")

        result = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode", "User likes Python"],
        )
        assert result == 1.0

    def test_recall_at_k_per_call_overrides_instance(self):
        """Per-call conversation_id should override instance default."""
        gta = _make_gta(conversation_id="conv_1")

        result = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers light mode", "User likes JavaScript"],
            conversation_id="conv_2",
        )
        assert result == 1.0

    def test_recall_at_k_no_conversation_id_unscoped(self):
        """Without conversation_id, all entries for that query are returned (legacy behavior)."""
        gta = _make_gta()

        result = gta.recall_at_k(
            "What happened yesterday?",
            ["Team standup at 9am"],
        )
        # Only 1 of 2 expected chunks matched
        assert result == 0.5

    def test_recall_at_k_partial_match(self):
        """Partial recall should compute correctly."""
        gta = _make_gta()

        result = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode"],  # only 1 of 2 expected
            conversation_id="conv_1",
        )
        assert result == 0.5

    def test_recall_at_k_no_ground_truth(self):
        """Query with no matching ground truth should return nan."""
        gta = _make_gta()

        result = gta.recall_at_k(
            "Nonexistent query",
            ["Some memory"],
            conversation_id="conv_1",
        )
        assert np.isnan(result)

    def test_recall_at_k_empty_retrieved(self):
        """Empty retrieved list should return 0.0."""
        gta = _make_gta()

        result = gta.recall_at_k(
            "What are the user's preferences?",
            [],
            conversation_id="conv_1",
        )
        assert result == 0.0

    def test_recall_at_k_with_k(self):
        """k parameter should limit retrieved chunks considered."""
        gta = _make_gta()

        result = gta.recall_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode", "User likes Python", "Irrelevant"],
            k=1,
            conversation_id="conv_1",
        )
        # Only top-1 considered, matched 1 of 2 expected
        assert result == 0.5


# ---------------------------------------------------------------------------
# precision_at_k with conversation_id
# ---------------------------------------------------------------------------


class TestPrecisionAtKWithConversationId:
    def test_precision_at_k_scoped_to_conversation(self):
        """Precision should only consider ground truth from the scoped conversation."""
        gta = _make_gta()

        result = gta.precision_at_k(
            "What are the user's preferences?",
            ["User prefers dark mode", "Irrelevant memory"],
            conversation_id="conv_1",
        )
        # 1 of 2 retrieved is in expected set
        assert result == 0.5

    def test_precision_at_k_no_ground_truth(self):
        gta = _make_gta()

        result = gta.precision_at_k(
            "Nonexistent query",
            ["Some memory"],
            conversation_id="conv_1",
        )
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# mrr with conversation_id
# ---------------------------------------------------------------------------


class TestMRRWithConversationId:
    def test_mrr_scoped_to_conversation(self):
        """MRR should find first relevant memory in the correct conversation."""
        gta = _make_gta()

        result = gta.mrr(
            "What are the user's preferences?",
            ["Irrelevant", "User prefers dark mode", "User likes Python"],
            conversation_id="conv_1",
        )
        # First relevant at position 2 -> 1/2 = 0.5
        assert result == 0.5

    def test_mrr_first_position(self):
        """MRR with relevant item at first position."""
        gta = _make_gta()

        result = gta.mrr(
            "What are the user's preferences?",
            ["User prefers dark mode"],
            conversation_id="conv_1",
        )
        assert result == 1.0

    def test_mrr_wrong_conversation(self):
        """MRR with no relevant items from the correct conversation."""
        gta = _make_gta()

        result = gta.mrr(
            "What are the user's preferences?",
            ["User prefers light mode", "User likes JavaScript"],
            conversation_id="conv_1",
        )
        assert result == 0.0

    def test_mrr_no_ground_truth(self):
        gta = _make_gta()

        result = gta.mrr(
            "Nonexistent query",
            ["Some memory"],
            conversation_id="conv_1",
        )
        assert np.isnan(result)
