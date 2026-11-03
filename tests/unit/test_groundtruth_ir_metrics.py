"""Regression tests for IR metric bounds on duplicate retrieved chunks.

recall_at_k counted occurrences of golden chunks in the retrieved list while
its denominator is the number of unique golden chunks, so a retriever that
returned the same relevant chunk twice produced recall > 1. ndcg_at_k credited
every duplicate retrieved position, pushing DCG above the ideal DCG and
yielding NDCG > 1. Both metrics are ratios bounded to [0, 1].
"""

from __future__ import annotations

import math
import unittest

from trulens.feedback.dummy.provider import DummyProvider
from trulens.feedback.groundtruth import GroundTruthAgreement


def _agreement(expected_chunks):
    return GroundTruthAgreement(
        ground_truth=[
            {
                "query": "q",
                "expected_response": "",
                "expected_chunks": expected_chunks,
            }
        ],
        provider=DummyProvider(),
    )


class TestIRMetricDuplicateChunks(unittest.TestCase):
    def test_recall_at_k_bounded_on_duplicate_relevant_chunk(self):
        a = _agreement([{"text": "A", "expect_score": 1}])
        self.assertEqual(a.recall_at_k("q", ["A", "A", "X"], k=3), 1.0)

    def test_recall_at_k_true_value_with_duplicates(self):
        a = _agreement([{"text": "A"}, {"text": "B"}])
        # One of two golden chunks retrieved (twice) -> recall 0.5, not 1.0.
        self.assertEqual(a.recall_at_k("q", ["A", "A"], k=2), 0.5)

    def test_recall_at_k_unchanged_without_duplicates(self):
        a = _agreement([{"text": "A"}, {"text": "B"}])
        self.assertEqual(a.recall_at_k("q", ["A", "B"], k=2), 1.0)

    def test_ndcg_at_k_bounded_on_duplicate_relevant_chunk(self):
        a = _agreement([{"text": "A", "expect_score": 1}])
        self.assertLessEqual(a.ndcg_at_k("q", ["A", "A"], k=2), 1.0)

    def test_ndcg_at_k_perfect_ranking_unchanged(self):
        a = _agreement([{"text": "A"}, {"text": "B"}])
        self.assertAlmostEqual(a.ndcg_at_k("q", ["A", "B"], k=2), 1.0)

    def test_ndcg_at_k_perfect_with_duplicate_golden_chunk(self):
        # The golden set counts each distinct chunk once, so retrieving A first is
        # a perfect ranking even when the same chunk is annotated twice.
        a = _agreement([
            {"text": "A", "expect_score": 1},
            {"text": "A", "expect_score": 1},
        ])
        self.assertAlmostEqual(a.ndcg_at_k("q", ["A", "A"], k=2), 1.0)

    def test_ndcg_at_k_perfect_with_partly_duplicated_golden_chunks(self):
        # A repeated chunk is one golden chunk, scored once, by its first annotation.
        a = _agreement([
            {"text": "A", "expect_score": 1},
            {"text": "A", "expect_score": 0.5},
        ])
        self.assertAlmostEqual(a.ndcg_at_k("q", ["A", "X"], k=2), 1.0)


class TestIRMetricZeroScoreGoldenChunks(unittest.TestCase):
    """A golden chunk with expect_score 0 was judged not relevant.

    ndcg_at_k gives it no gain; the other IR metrics counted it as relevant,
    so retrieving only that chunk scored a perfect hit rate, MRR and precision.
    """

    def setUp(self):
        self.a = _agreement([
            {"text": "A", "expect_score": 1},
            {"text": "B", "expect_score": 0},
        ])

    def test_retrieving_only_the_irrelevant_chunk_is_a_miss(self):
        self.assertEqual(self.a.ndcg_at_k("q", ["B"], k=1), 0.0)
        self.assertEqual(self.a.ir_hit_rate("q", ["B"], k=1), 0.0)
        self.assertEqual(self.a.mrr("q", ["B"]), 0.0)
        self.assertEqual(self.a.precision_at_k("q", ["B"], k=1), 0.0)

    def test_retrieving_the_only_relevant_chunk_is_full_recall(self):
        self.assertEqual(self.a.recall_at_k("q", ["A"], k=1), 1.0)

    def test_relevant_chunk_ranked_after_the_irrelevant_one(self):
        self.assertEqual(self.a.mrr("q", ["B", "A"]), 0.5)
        self.assertEqual(self.a.precision_at_k("q", ["B", "A"], k=2), 0.5)

    def test_only_irrelevant_golden_chunks_is_undefined(self):
        a = _agreement([{"text": "B", "expect_score": 0}])
        self.assertTrue(math.isnan(a.recall_at_k("q", ["B"], k=1)))
        self.assertTrue(math.isnan(a.mrr("q", ["B"])))
        self.assertTrue(math.isnan(a.ir_hit_rate("q", ["B"], k=1)))


if __name__ == "__main__":
    unittest.main()
