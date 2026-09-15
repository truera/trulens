"""Regression tests for IR metric bounds on duplicate retrieved chunks.

recall_at_k counted occurrences of golden chunks in the retrieved list while
its denominator is the number of unique golden chunks, so a retriever that
returned the same relevant chunk twice produced recall > 1. ndcg_at_k credited
every duplicate retrieved position, pushing DCG above the ideal DCG and
yielding NDCG > 1. Both metrics are ratios bounded to [0, 1].
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
