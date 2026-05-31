"""Tests for GroundTruthAgreement memory recall methods."""

import numpy as np
import pytest

from trulens.feedback.dummy.provider import DummyProvider
from trulens.feedback.groundtruth import GroundTruthAgreement


@pytest.fixture
def golden_set():
    return [
        {
            "query": "What are the user's preferences?",
            "expected_memories": [
                "User prefers dark mode",
                "User likes Python",
                "User works remotely",
            ],
            "conversation_id": "conv_123",
        },
        {
            "query": "Why was the project delayed?",
            "expected_memories": [
                "Team lead resigned in February",
                "API vendor changed pricing mid-contract",
            ],
            "conversation_id": "conv_123",
        },
        {
            "query": "What decisions were made?",
            "expected_memories": [
                "Team agreed to use React",
                "Deadline extended by two weeks",
            ],
            "conversation_id": "conv_456",
        },
    ]


@pytest.fixture
def provider():
    return DummyProvider(name="test_provider", delay=0.0)


@pytest.fixture
def gta(golden_set, provider):
    return GroundTruthAgreement(
        golden_set,
        provider=provider,
        conversation_id="conv_123",
    )


@pytest.fixture
def gta_no_conv(golden_set, provider):
    return GroundTruthAgreement(
        golden_set,
        provider=provider,
    )


# ---- memory_recall ----


class TestMemoryRecall:
    def test_perfect_recall(self, gta):
        result = gta.memory_recall(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "User prefers dark mode",
                "User likes Python",
                "User works remotely",
            ],
        )
        assert result == 1.0

    def test_partial_recall(self, gta):
        result = gta.memory_recall(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "User prefers dark mode",
                "User works remotely",
            ],
        )
        assert result == pytest.approx(2 / 3, abs=1e-6)

    def test_zero_recall(self, gta):
        result = gta.memory_recall(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "Team agreed to use React",
                "Deadline extended by two weeks",
            ],
        )
        assert result == 0.0

    def test_empty_retrieved(self, gta):
        result = gta.memory_recall(
            prompt="What are the user's preferences?",
            retrieved_memories=[],
        )
        assert result == 0.0

    def test_no_ground_truth_wrong_conv(self, gta):
        result = gta.memory_recall(
            prompt="What decisions were made?",
            retrieved_memories=["Team agreed to use React"],
        )
        # conv_456 doesn't match gta's conversation_id="conv_123"
        assert np.isnan(result)

    def test_no_ground_truth_no_conv_filter(self, gta_no_conv):
        result = gta_no_conv.memory_recall(
            prompt="What decisions were made?",
            retrieved_memories=["Team agreed to use React"],
        )
        # Without conversation_id filter, should find conv_456's entry
        assert result == 0.5

    def test_unknown_query(self, gta):
        result = gta.memory_recall(
            prompt="What is the weather?",
            retrieved_memories=["It is sunny"],
        )
        assert np.isnan(result)


# ---- memory_mrr ----


class TestMemoryMRR:
    def test_relevant_at_rank_1(self, gta):
        result = gta.memory_mrr(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "User prefers dark mode",
                "Irrelevant memory",
            ],
        )
        assert result == 1.0

    def test_relevant_at_rank_3(self, gta):
        result = gta.memory_mrr(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "Irrelevant 1",
                "Irrelevant 2",
                "User prefers dark mode",
            ],
        )
        assert result == pytest.approx(1 / 3, abs=1e-6)

    def test_no_relevant(self, gta):
        result = gta.memory_mrr(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "Irrelevant 1",
                "Irrelevant 2",
            ],
        )
        assert result == 0.0

    def test_empty_retrieved(self, gta):
        result = gta.memory_mrr(
            prompt="What are the user's preferences?",
            retrieved_memories=[],
        )
        assert result == 0.0

    def test_no_ground_truth(self, gta):
        result = gta.memory_mrr(
            prompt="Unknown query",
            retrieved_memories=["Some memory"],
        )
        assert np.isnan(result)


# ---- _is_similar ----


class TestIsSimilar:
    def test_exact_match(self):
        assert GroundTruthAgreement._is_similar("hello world", "hello world", 1.0)

    def test_exact_no_match(self):
        assert not GroundTruthAgreement._is_similar("hello world", "hello earth", 1.0)

    def test_fuzzy_match_high_similarity(self):
        assert GroundTruthAgreement._is_similar(
            "User prefers dark mode", "User prefers dark mode.", 0.9
        )

    def test_fuzzy_no_match_low_similarity(self):
        assert not GroundTruthAgreement._is_similar(
            "User prefers dark mode", "Team agreed to use React", 0.5
        )


# ---- fuzzy matching integration ----


class TestFuzzyMatching:
    def test_fuzzy_recall(self, gta):
        result = gta.memory_recall(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "User prefers dark mode.",  # trailing period
                "User likes Python programming",  # extra word
                "User works remotely from home",  # extra words
            ],
            similarity_threshold=0.7,
        )
        # All should fuzzy match with threshold 0.7
        assert result > 0.5

    def test_fuzzy_mrr(self, gta):
        result = gta.memory_mrr(
            prompt="What are the user's preferences?",
            retrieved_memories=[
                "User prefers dark mode.",
            ],
            similarity_threshold=0.8,
        )
        assert result == 1.0


# ---- conversation_id filtering ----


class TestConversationId:
    def test_conv_filter_blocks_other_conv(self, golden_set, provider):
        gta = GroundTruthAgreement(
            golden_set,
            provider=provider,
            conversation_id="conv_123",
        )
        result = gta.memory_recall(
            prompt="What decisions were made?",
            retrieved_memories=["Team agreed to use React"],
        )
        assert np.isnan(result)

    def test_conv_filter_allows_matching_conv(self, golden_set, provider):
        gta = GroundTruthAgreement(
            golden_set,
            provider=provider,
            conversation_id="conv_456",
        )
        result = gta.memory_recall(
            prompt="What decisions were made?",
            retrieved_memories=["Team agreed to use React"],
        )
        assert result == 0.5

    def test_no_conv_filter_searches_all(self, golden_set, provider):
        gta = GroundTruthAgreement(
            golden_set,
            provider=provider,
        )
        result = gta.memory_recall(
            prompt="What decisions were made?",
            retrieved_memories=["Team agreed to use React"],
        )
        assert result == 0.5
