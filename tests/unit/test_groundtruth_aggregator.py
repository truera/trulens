"""Tests for GroundTruthAggregator class."""

from unittest import TestCase

import numpy as np
import pytest
from trulens.feedback.groundtruth import GroundTruthAggregator


class TestGroundTruthAggregator(TestCase):
    """Tests for GroundTruthAggregator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.true_labels = [1, 0, 1, 0, 1]
        self.aggregator = GroundTruthAggregator(true_labels=self.true_labels)
        self.perfect_scores = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.worst_scores = [0.0, 1.0, 0.0, 1.0, 0.0]
        # Use simple test data that results in exact calculations
        # scores = [0.5, 0.0, 1.0, 0.0, 0.5] with labels [1, 0, 1, 0, 1]
        # BS = [(0.5-1)^2 + (0.0-0)^2 + (1.0-1)^2 + (0.0-0)^2 + (0.5-1)^2] / 5
        #    = [0.25 + 0 + 0 + 0 + 0.25] / 5 = 0.5 / 5 = 0.1 (exact)
        self.test_scores = [0.5, 0.0, 1.0, 0.0, 0.5]

    def _calculate_expected_brier_score(self, scores, labels):
        """Helper to calculate expected Brier score."""
        if not scores:
            return np.nan
        return np.mean([
            (score - label) ** 2 for score, label in zip(scores, labels)
        ])

    @pytest.mark.optional
    def test_brier_score_basic_functionality(self):
        """Test basic Brier score functionality."""
        # Test with simple scores that result in exact calculation
        result = self.aggregator.brier_score(self.test_scores)
        self.assertEqual(result, 0.1)  # Exact calculation: 0.5 / 5 = 0.1

        # Test with score-confidence pairs (should use only first element)
        # Convert to [score, confidence] format
        score_confidence_pairs = [
            [0.5, 0.9],
            [0.0, 0.7],
            [1.0, 0.8],
            [0.0, 0.6],
            [0.5, 0.85],
        ]
        result = self.aggregator.brier_score(score_confidence_pairs)
        self.assertEqual(result, 0.1)  # Same exact calculation

    @pytest.mark.optional
    def test_brier_score_edge_cases(self):
        """Test Brier score with edge cases."""
        # Empty scores
        result = self.aggregator.brier_score([])
        self.assertTrue(np.isnan(result))

        # Perfect predictions: BS = 0.0
        result = self.aggregator.brier_score(self.perfect_scores)
        self.assertEqual(result, 0.0)  # Exact calculation

        # Worst predictions: BS = 1.0
        result = self.aggregator.brier_score(self.worst_scores)
        self.assertEqual(result, 1.0)  # Exact calculation

        # Single score: (0.5 - 1)^2 = 0.25
        single_aggregator = GroundTruthAggregator(true_labels=[1])
        result = single_aggregator.brier_score([0.5])
        self.assertEqual(result, 0.25)  # Exact calculation

    @pytest.mark.optional
    def test_brier_score_validation(self):
        """Test Brier score validation and error handling."""
        # Length mismatch
        with self.assertRaises(AssertionError):
            self.aggregator.brier_score([
                0.8,
                0.3,
                0.9,
            ])  # Only 3 scores for 5 labels

        # Empty aggregator with empty scores
        empty_aggregator = GroundTruthAggregator(true_labels=[])
        result = empty_aggregator.brier_score([])
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_brier_score_boundary_values(self):
        """Test Brier score with boundary probability values."""
        # Test with edge probabilities: [0.0, 0.5, 1.0, 0.5, 0.0] with labels [1, 0, 1, 0, 1]
        # BS = [(0.0-1)^2 + (0.5-0)^2 + (1.0-1)^2 + (0.5-0)^2 + (0.0-1)^2] / 5
        #    = [1 + 0.25 + 0 + 0.25 + 1] / 5 = 2.5 / 5 = 0.5 (exact)
        edge_scores = [0.0, 0.5, 1.0, 0.5, 0.0]
        result = self.aggregator.brier_score(edge_scores)
        self.assertEqual(result, 0.5)  # Exact calculation

        # Verify score is within valid range [0, 1]
        result = self.aggregator.brier_score(self.test_scores)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
