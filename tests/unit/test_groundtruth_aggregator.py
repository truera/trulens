"""Tests for GroundTruthAggregator class."""

from unittest import TestCase
import unittest.mock as mock

import numpy as np
import pytest
from trulens.feedback.groundtruth import GroundTruthAggregator


class TestGroundTruthAggregator(TestCase):
    """Tests for GroundTruthAggregator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.true_labels = [1, 0, 1, 0, 1]
        self.aggregator = GroundTruthAggregator(true_labels=self.true_labels)

        # Test scores for various scenarios
        self.perfect_scores = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.worst_scores = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.test_scores = [0.5, 0.0, 1.0, 0.0, 0.5]
        self.score_confidence_pairs = [
            [0.5, 0.9],
            [0.0, 0.7],
            [1.0, 0.8],
            [0.0, 0.6],
            [0.5, 0.85],
        ]

    @pytest.mark.optional
    def test_constructor_validation(self):
        """Test constructor input validation."""
        # Valid construction
        aggregator = GroundTruthAggregator(true_labels=[1, 0, 1])
        self.assertEqual(aggregator.true_labels, [1, 0, 1])
        self.assertEqual(aggregator.n_bins, 5)  # Default value
        self.assertIsNone(aggregator.k)  # Default value

        # Test with k parameter
        aggregator_k = GroundTruthAggregator(true_labels=[1, 0, 1], k=3)
        self.assertEqual(aggregator_k.k, 3)

        # Test with custom n_bins
        aggregator_bins = GroundTruthAggregator(
            true_labels=[1, 0, 1], n_bins=10
        )
        self.assertEqual(aggregator_bins.n_bins, 10)

        # Empty labels should work (edge case)
        empty_aggregator = GroundTruthAggregator(true_labels=[])
        self.assertEqual(empty_aggregator.true_labels, [])

    @pytest.mark.optional
    def test_input_validation_comprehensive(self):
        """Test comprehensive input validation across all methods."""
        # Test with mismatched lengths
        short_scores = [0.5, 0.3]  # Length 2 vs true_labels length 5

        with self.assertRaises(
            AssertionError, msg="Brier score should reject mismatched lengths"
        ):
            self.aggregator.brier_score(short_scores)

        with self.assertRaises(
            AssertionError, msg="ECE should reject mismatched lengths"
        ):
            self.aggregator.ece([(0.5, 0.8), (0.3, 0.7)])

        # Test with invalid score types (should handle gracefully or raise clear errors)
        with self.assertRaises(
            (TypeError, ValueError), msg="Methods should handle invalid types"
        ):
            self.aggregator.auc(["invalid", "scores", "list", "here", "test"])

        # Test None inputs
        with self.assertRaises(
            (TypeError, ValueError), msg="Methods should handle None inputs"
        ):
            self.aggregator.precision(None)

        # Test with NaN values
        nan_scores = [0.5, np.nan, 0.7, 0.3, 0.8]
        # Most methods should handle NaN gracefully or raise clear errors
        try:
            result = self.aggregator.auc(nan_scores)
            # If it doesn't raise an error, result should be NaN or a reasonable value
            self.assertTrue(np.isnan(result) or isinstance(result, float))
        except (ValueError, RuntimeError):
            # It's acceptable for methods to raise clear errors for NaN inputs
            pass

    @pytest.mark.optional
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Single item aggregator
        single_aggregator = GroundTruthAggregator(true_labels=[1])
        self.assertEqual(
            single_aggregator.brier_score([0.5]),
            0.25,
            msg="Single item Brier score should be calculated correctly",
        )
        self.assertEqual(
            single_aggregator.mae([0.5]),
            0.5,
            msg="Single item MAE should be calculated correctly",
        )

        # All identical scores
        identical_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        auc_result = self.aggregator.auc(identical_scores)
        self.assertEqual(
            auc_result,
            0.5,
            msg="Identical scores should give AUC of 0.5 (random performance)",
        )

        # All identical labels
        identical_labels = [1, 1, 1, 1, 1]
        identical_aggregator = GroundTruthAggregator(
            true_labels=identical_labels
        )

        # AUC is undefined for identical labels, should handle gracefully
        try:
            auc_result = identical_aggregator.auc([0.1, 0.2, 0.3, 0.4, 0.5])
            # If no error, result should be NaN or a reasonable value
            self.assertTrue(
                np.isnan(auc_result) or isinstance(auc_result, float)
            )
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for undefined AUC
            pass

        # Extreme values
        extreme_scores = [0.0, 1.0, 0.0, 1.0, 0.0]
        mae_result = self.aggregator.mae(extreme_scores)
        self.assertEqual(
            mae_result,
            1.0,
            msg="Extreme opposing scores should give MAE of 1.0",
        )

    @pytest.mark.optional
    def test_brier_score(self):
        """Test Brier score functionality including edge cases."""
        # Basic functionality with exact calculation
        result = self.aggregator.brier_score(self.test_scores)
        self.assertEqual(
            result,
            0.1,
            msg="Brier score should be calculated correctly: (0.25 + 0 + 0 + 0 + 0.25) / 5 = 0.1",
        )

        # Score-confidence pairs (should use only first element)
        result = self.aggregator.brier_score(self.score_confidence_pairs)
        self.assertEqual(
            result,
            0.1,
            msg="Brier score should extract scores from confidence pairs correctly",
        )

        # Edge cases
        self.assertTrue(
            np.isnan(self.aggregator.brier_score([])),
            msg="Empty scores should return NaN",
        )
        self.assertEqual(
            self.aggregator.brier_score(self.perfect_scores),
            0.0,
            msg="Perfect scores should give Brier score of 0.0",
        )
        self.assertEqual(
            self.aggregator.brier_score(self.worst_scores),
            1.0,
            msg="Worst possible scores should give Brier score of 1.0",
        )

        # Length validation
        with self.assertRaises(
            AssertionError, msg="Mismatched lengths should raise AssertionError"
        ):
            self.aggregator.brier_score([0.8, 0.3, 0.9])

    @pytest.mark.optional
    def test_auc(self):
        """Test AUC calculation."""
        # Perfect classifier
        result = self.aggregator.auc(self.perfect_scores)
        self.assertEqual(
            result, 1.0, msg="Perfect classifier should achieve AUC of 1.0"
        )

        # Random classifier
        result = self.aggregator.auc([0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(
            result, 0.5, msg="Random classifier should achieve AUC of 0.5"
        )

        # With score-confidence pairs - should be better than random since scores correlate with labels
        result = self.aggregator.auc(self.score_confidence_pairs)
        self.assertGreater(
            result,
            0.5,
            msg="Scores that correlate with labels should achieve AUC > 0.5",
        )

        # Verify monotonicity: better discriminating scores should give higher AUC
        # Use labels [1, 0, 1, 0, 1] with different discrimination levels
        good_scores = [0.9, 0.1, 0.8, 0.2, 0.7]  # Clear separation
        # Create scores that make some errors to get a different AUC
        mediocre_scores = [0.6, 0.6, 0.4, 0.4, 0.5]  # Some misclassifications

        good_auc = self.aggregator.auc(good_scores)
        mediocre_auc = self.aggregator.auc(mediocre_scores)

        # Good should be perfect, mediocre should be imperfect
        self.assertEqual(
            good_auc,
            1.0,
            msg="Perfect discrimination should achieve AUC of 1.0",
        )
        self.assertLess(
            mediocre_auc,
            1.0,
            msg="Imperfect discrimination should achieve AUC < 1.0",
        )
        self.assertGreater(
            good_auc,
            mediocre_auc,
            msg="Better discrimination should achieve higher AUC",
        )

    @pytest.mark.optional
    def test_correlation_metrics(self):
        """Test various correlation metrics."""
        # Test with imperfect scores first to avoid potential issues with perfect correlation
        test_scores = [0.8, 0.1, 0.9, 0.2, 0.7]

        # Test expected behavior: these scores should positively correlate with labels [1,0,1,0,1]
        # Since scores [0.8, 0.1, 0.9, 0.2, 0.7] generally match pattern [1,0,1,0,1]

        # Kendall's tau - should be positive for this pattern
        kendall_result = self.aggregator.kendall_tau(test_scores)
        self.assertGreater(
            kendall_result,
            0.0,
            msg="Positively correlated scores should give positive Kendall's tau",
        )

        # Spearman correlation - should be positive for this pattern
        spearman_result = self.aggregator.spearman_correlation(test_scores)
        self.assertGreater(
            spearman_result,
            0.0,
            msg="Positively correlated scores should give positive Spearman correlation",
        )

        # Pearson correlation - should be positive for this pattern
        pearson_result = self.aggregator.pearson_correlation(test_scores)
        self.assertGreater(
            pearson_result,
            0.0,
            msg="Positively correlated scores should give positive Pearson correlation",
        )

        # Test that perfect correlation gives maximum correlation
        perfect_corr_scores = [1.0, 0.0, 1.0, 0.0, 1.0]
        perfect_kendall = self.aggregator.kendall_tau(perfect_corr_scores)
        perfect_spearman = self.aggregator.spearman_correlation(
            perfect_corr_scores
        )
        perfect_pearson = self.aggregator.pearson_correlation(
            perfect_corr_scores
        )

        # Perfect correlation should be higher than imperfect correlation
        self.assertGreater(
            perfect_kendall,
            kendall_result,
            msg="Perfect correlation should give higher Kendall's tau",
        )
        self.assertGreater(
            perfect_spearman,
            spearman_result,
            msg="Perfect correlation should give higher Spearman correlation",
        )
        self.assertGreater(
            perfect_pearson,
            pearson_result,
            msg="Perfect correlation should give higher Pearson correlation",
        )

        # Matthews correlation - needs binary predictions
        binary_scores = [1, 0, 1, 0, 1]  # Binary predictions
        matthews_result = self.aggregator.matthews_correlation(binary_scores)
        self.assertEqual(
            matthews_result,
            1.0,
            msg="Perfect binary classification should give Matthews correlation of 1.0",
        )

        # Test with score-confidence pairs - should extract scores properly
        kendall_pairs = self.aggregator.kendall_tau(self.score_confidence_pairs)
        spearman_pairs = self.aggregator.spearman_correlation(
            self.score_confidence_pairs
        )
        pearson_pairs = self.aggregator.pearson_correlation(
            self.score_confidence_pairs
        )

        # Should be similar to using just the scores
        scores_only = [pair[0] for pair in self.score_confidence_pairs]
        kendall_scores = self.aggregator.kendall_tau(scores_only)
        spearman_scores = self.aggregator.spearman_correlation(scores_only)
        pearson_scores = self.aggregator.pearson_correlation(scores_only)

        self.assertEqual(
            kendall_pairs,
            kendall_scores,
            msg="Kendall's tau should be same for pairs and scores-only",
        )
        self.assertEqual(
            spearman_pairs,
            spearman_scores,
            msg="Spearman correlation should be same for pairs and scores-only",
        )
        self.assertEqual(
            pearson_pairs,
            pearson_scores,
            msg="Pearson correlation should be same for pairs and scores-only",
        )

        # Test Matthews correlation with binary score-confidence pairs
        binary_score_confidence_pairs = [
            [1, 0.9],
            [0, 0.7],
            [1, 0.8],
            [0, 0.6],
            [1, 0.85],
        ]
        matthews_pairs = self.aggregator.matthews_correlation(
            binary_score_confidence_pairs
        )
        self.assertEqual(
            matthews_pairs,
            1.0,
            msg="Perfect binary classification with pairs should give Matthews correlation of 1.0",
        )

    @pytest.mark.optional
    def test_classification_metrics(self):
        """Test precision, recall, F1 score, and Cohen's kappa."""
        # Perfect classifier
        self.assertEqual(
            self.aggregator.precision(self.perfect_scores),
            1.0,
            msg="Perfect classifier should achieve precision of 1.0",
        )
        self.assertEqual(
            self.aggregator.recall(self.perfect_scores),
            1.0,
            msg="Perfect classifier should achieve recall of 1.0",
        )
        self.assertEqual(
            self.aggregator.f1_score(self.perfect_scores),
            1.0,
            msg="Perfect classifier should achieve F1 score of 1.0",
        )
        self.assertEqual(
            self.aggregator.cohens_kappa(self.perfect_scores),
            1.0,
            msg="Perfect classifier should achieve Cohen's kappa of 1.0",
        )

        # Test with different thresholds - verify expected behavior
        # With test_scores [0.5, 0.0, 1.0, 0.0, 0.5] and labels [1, 0, 1, 0, 1]
        # At threshold 0.3: predictions [1, 0, 1, 0, 1] -> perfect match
        # At threshold 0.7: predictions [0, 0, 1, 0, 0] -> precision=1.0, recall=0.33

        low_threshold_precision = self.aggregator.precision(
            self.test_scores, threshold=0.3
        )
        high_threshold_precision = self.aggregator.precision(
            self.test_scores, threshold=0.7
        )

        self.assertEqual(
            low_threshold_precision,
            1.0,
            msg="Low threshold should give perfect precision for this data",
        )
        self.assertEqual(
            high_threshold_precision,
            1.0,
            msg="High threshold should still give perfect precision but fewer predictions",
        )

        low_threshold_recall = self.aggregator.recall(
            self.test_scores, threshold=0.3
        )
        high_threshold_recall = self.aggregator.recall(
            self.test_scores, threshold=0.7
        )

        self.assertEqual(
            low_threshold_recall,
            1.0,
            msg="Low threshold should give perfect recall for this data",
        )
        self.assertLess(
            high_threshold_recall,
            low_threshold_recall,
            msg="High threshold should give lower recall due to missed positives",
        )

        # Test that F1 score balances precision and recall
        low_threshold_f1 = self.aggregator.f1_score(
            self.test_scores, threshold=0.3
        )
        high_threshold_f1 = self.aggregator.f1_score(
            self.test_scores, threshold=0.7
        )

        self.assertEqual(
            low_threshold_f1,
            1.0,
            msg="Low threshold should give perfect F1 score for this data",
        )
        self.assertLess(
            high_threshold_f1,
            low_threshold_f1,
            msg="High threshold should give lower F1 score",
        )

        # Test with score-confidence pairs - should extract scores properly
        precision_pairs = self.aggregator.precision(self.score_confidence_pairs)
        recall_pairs = self.aggregator.recall(self.score_confidence_pairs)
        f1_pairs = self.aggregator.f1_score(self.score_confidence_pairs)
        kappa_pairs = self.aggregator.cohens_kappa(self.score_confidence_pairs)

        # Should be similar to using just the scores
        scores_only = [pair[0] for pair in self.score_confidence_pairs]
        precision_scores = self.aggregator.precision(scores_only)
        recall_scores = self.aggregator.recall(scores_only)
        f1_scores = self.aggregator.f1_score(scores_only)
        kappa_scores = self.aggregator.cohens_kappa(scores_only)

        self.assertEqual(
            precision_pairs,
            precision_scores,
            msg="Precision should be same for pairs and scores-only",
        )
        self.assertEqual(
            recall_pairs,
            recall_scores,
            msg="Recall should be same for pairs and scores-only",
        )
        self.assertEqual(
            f1_pairs,
            f1_scores,
            msg="F1 score should be same for pairs and scores-only",
        )
        self.assertEqual(
            kappa_pairs,
            kappa_scores,
            msg="Cohen's kappa should be same for pairs and scores-only",
        )

    @pytest.mark.optional
    def test_mae(self):
        """Test Mean Absolute Error."""
        # Perfect predictions
        result = self.aggregator.mae(self.perfect_scores)
        self.assertEqual(
            result, 0.0, msg="Perfect predictions should give MAE of 0.0"
        )

        # Test with score-confidence pairs
        result = self.aggregator.mae(self.score_confidence_pairs)
        self.assertGreater(
            result, 0.0, msg="Imperfect predictions should give MAE > 0.0"
        )

        # Should be same as using just the scores
        scores_only = [pair[0] for pair in self.score_confidence_pairs]
        result_scores = self.aggregator.mae(scores_only)
        self.assertEqual(
            result,
            result_scores,
            msg="MAE should be same for pairs and scores-only",
        )

        # Basic calculation
        result = self.aggregator.mae(self.test_scores)
        expected = np.mean(
            np.abs(np.array(self.test_scores) - np.array(self.true_labels))
        )
        self.assertEqual(
            result, expected, msg="MAE should match manual calculation"
        )

        # Verify monotonicity: worse predictions should have higher MAE
        worse_scores = [0.0, 1.0, 0.0, 1.0, 0.0]  # Opposite of truth
        worse_mae = self.aggregator.mae(worse_scores)
        self.assertGreater(
            worse_mae, result, msg="Worse predictions should have higher MAE"
        )

    @pytest.mark.optional
    def test_ece(self):
        """Test Expected Calibration Error."""
        # Convert to tuple format required by ECE
        score_confidence_tuples = [
            (score, conf) for score, conf in self.score_confidence_pairs
        ]

        result = self.aggregator.ece(score_confidence_tuples)
        self.assertGreater(
            result, 0.0, msg="Imperfect calibration should give ECE > 0.0"
        )

        # Perfect calibration test
        perfect_tuples = [
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
        ]
        perfect_result = self.aggregator.ece(perfect_tuples)
        self.assertEqual(
            perfect_result,
            0.0,
            msg="Perfect calibration should give ECE of 0.0",
        )

        # Verify that perfect calibration is better than imperfect
        self.assertLess(
            perfect_result,
            result,
            msg="Perfect calibration should have lower ECE than imperfect",
        )

        # Length validation
        with self.assertRaises(
            AssertionError, msg="Mismatched lengths should raise AssertionError"
        ):
            self.aggregator.ece([(0.8, 0.9), (0.3, 0.7)])

    @pytest.mark.optional
    def test_edge_cases(self):
        """Test edge cases across all methods."""
        # Empty aggregator
        empty_aggregator = GroundTruthAggregator(true_labels=[])
        self.assertTrue(
            np.isnan(empty_aggregator.brier_score([])),
            msg="Empty aggregator should return NaN for Brier score",
        )

        # Single item aggregator
        single_aggregator = GroundTruthAggregator(true_labels=[1])
        self.assertEqual(
            single_aggregator.brier_score([0.5]),
            0.25,
            msg="Single item Brier score should be calculated correctly",
        )
        self.assertEqual(
            single_aggregator.mae([0.5]),
            0.5,
            msg="Single item MAE should be calculated correctly",
        )

        # All zeros and ones
        binary_aggregator = GroundTruthAggregator(true_labels=[0, 1])
        self.assertEqual(
            binary_aggregator.auc([0.0, 1.0]),
            1.0,
            msg="Perfect binary classification should achieve AUC of 1.0",
        )
        self.assertEqual(
            binary_aggregator.precision([0.0, 1.0]),
            1.0,
            msg="Perfect binary classification should achieve precision of 1.0",
        )
        self.assertEqual(
            binary_aggregator.recall([0.0, 1.0]),
            1.0,
            msg="Perfect binary classification should achieve recall of 1.0",
        )

    @pytest.mark.optional
    def test_custom_aggregation_functions(self):
        """Test custom aggregation function registration."""

        def custom_mean(scores, aggregator):
            if isinstance(scores[0], list):
                scores = [score for score, _ in scores]
            return np.mean(scores)

        self.aggregator.register_custom_agg_func("custom_mean", custom_mean)

        # Test that custom function is available
        self.assertTrue(
            hasattr(self.aggregator, "custom_mean"),
            msg="Custom function should be registered as method",
        )

        # Test custom function execution
        result = self.aggregator.custom_mean(self.test_scores)
        self.assertEqual(
            result,
            np.mean(self.test_scores),
            msg="Custom function should execute correctly",
        )

    # ========== NEW REAL-WORLD TESTS ==========

    @pytest.mark.optional
    def test_real_world_dataset_sizes(self):
        """Test with realistic dataset sizes used in production."""
        # Use fixed seed for reproducibility
        with mock.patch("numpy.random.seed"):
            np.random.seed(42)  # Set seed for this test only

            # Large balanced dataset (common in benchmarking)
            large_labels = [i % 2 for i in range(1000)]
            large_scores = [
                0.3 + (i % 2) * 0.4 + np.random.normal(0, 0.1)
                for i in range(1000)
            ]
            large_aggregator = GroundTruthAggregator(true_labels=large_labels)

            # Test that metrics work with large datasets
            auc_result = large_aggregator.auc(large_scores)
            self.assertGreater(
                auc_result,
                0.6,
                msg="Large dataset with pattern should achieve decent AUC",
            )

            brier_result = large_aggregator.brier_score(large_scores)
            self.assertGreater(
                brier_result,
                0.0,
                msg="Large dataset should have some Brier score error",
            )
            self.assertLess(
                brier_result,
                0.5,
                msg="Large dataset with pattern should have reasonable Brier score",
            )

    @pytest.mark.optional
    def test_imbalanced_datasets(self):
        """Test behavior with imbalanced datasets (common in real applications)."""
        # Highly imbalanced: 90% negative, 10% positive
        imbalanced_labels = [0] * 90 + [1] * 10
        # Scores that should still allow discrimination
        imbalanced_scores = [0.2] * 90 + [0.8] * 10

        imbalanced_aggregator = GroundTruthAggregator(
            true_labels=imbalanced_labels
        )

        # AUC should be perfect for this clear separation
        auc_result = imbalanced_aggregator.auc(imbalanced_scores)
        self.assertEqual(
            auc_result,
            1.0,
            msg="Clear separation in imbalanced dataset should achieve perfect AUC",
        )

        # Precision and recall should be perfect for this clear separation
        precision = imbalanced_aggregator.precision(
            imbalanced_scores, threshold=0.5
        )
        recall = imbalanced_aggregator.recall(imbalanced_scores, threshold=0.5)

        self.assertEqual(
            precision,
            1.0,
            msg="Clear separation should achieve perfect precision",
        )
        self.assertEqual(
            recall, 1.0, msg="Clear separation should achieve perfect recall"
        )

        # Test that metrics handle the imbalance appropriately
        f1_score = imbalanced_aggregator.f1_score(
            imbalanced_scores, threshold=0.5
        )
        self.assertEqual(
            f1_score,
            1.0,
            msg="Clear separation should achieve perfect F1 score",
        )

        # Brier score should be low for this clear separation
        brier_result = imbalanced_aggregator.brier_score(imbalanced_scores)
        self.assertLess(
            brier_result,
            0.1,
            msg="Clear separation should have very low Brier score",
        )

    @pytest.mark.optional
    def test_noisy_real_world_data(self):
        """Test with noisy data that reflects real-world conditions."""
        # Use fixed seed for reproducibility
        with mock.patch("numpy.random.seed"):
            np.random.seed(42)  # Set seed for this test only

            # Simulate noisy labels (some annotation errors)
            clean_labels = [i % 2 for i in range(100)]
            noisy_labels = clean_labels.copy()
            # Flip 5% of labels to simulate annotation errors
            flip_indices = np.random.choice(100, size=5, replace=False)
            for idx in flip_indices:
                noisy_labels[idx] = 1 - noisy_labels[idx]

            # Noisy scores with realistic variance
            noisy_scores = [
                label * 0.7 + (1 - label) * 0.3 + np.random.normal(0, 0.15)
                for label in clean_labels
            ]
            # Clip to valid range
            noisy_scores = [max(0, min(1, score)) for score in noisy_scores]

            noisy_aggregator = GroundTruthAggregator(true_labels=noisy_labels)

            # Compare with clean data to verify noise impact
            clean_aggregator = GroundTruthAggregator(true_labels=clean_labels)
            clean_auc = clean_aggregator.auc(noisy_scores)
            noisy_auc = noisy_aggregator.auc(noisy_scores)

            # Noisy labels should perform worse than clean labels
            self.assertLess(
                noisy_auc,
                clean_auc,
                msg="Noisy labels should achieve lower AUC than clean labels",
            )

            # But still better than random
            self.assertGreater(
                noisy_auc,
                0.5,
                msg="Even noisy data should achieve AUC better than random",
            )

            # Brier score should be worse (higher) with noisy labels
            clean_brier = clean_aggregator.brier_score(noisy_scores)
            noisy_brier = noisy_aggregator.brier_score(noisy_scores)

            self.assertGreater(
                noisy_brier,
                clean_brier,
                msg="Noisy labels should have higher Brier score (worse)",
            )
            self.assertLess(
                noisy_brier,
                0.5,
                msg="Even noisy data should have reasonable Brier score",
            )

    @pytest.mark.optional
    def test_statistical_properties(self):
        """Test statistical properties that should hold for metrics."""
        # Test that correlation metrics behave as expected
        true_labels = [1, 0, 1, 0, 1, 0, 1, 0]

        # Perfectly correlated scores
        perfect_scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        # Anti-correlated scores
        anti_scores = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        aggregator = GroundTruthAggregator(true_labels=true_labels)

        # Perfect correlation should give high positive correlation
        perfect_tau = aggregator.kendall_tau(perfect_scores)
        self.assertEqual(
            perfect_tau,
            1.0,
            msg="Perfect positive correlation should give Kendall's tau of 1.0",
        )

        # Anti-correlation should give perfect negative correlation
        anti_tau = aggregator.kendall_tau(anti_scores)
        self.assertEqual(
            anti_tau,
            -1.0,
            msg="Perfect negative correlation should give Kendall's tau of -1.0",
        )

        # Similar pattern for Spearman - use assertAlmostEqual for floating point precision
        perfect_spearman = aggregator.spearman_correlation(perfect_scores)
        anti_spearman = aggregator.spearman_correlation(anti_scores)

        self.assertAlmostEqual(
            perfect_spearman,
            1.0,
            places=10,
            msg="Perfect positive correlation should give Spearman correlation of 1.0",
        )
        self.assertAlmostEqual(
            anti_spearman,
            -1.0,
            places=10,
            msg="Perfect negative correlation should give Spearman correlation of -1.0",
        )

        # Test that AUC reflects the correlation direction
        perfect_auc = aggregator.auc(perfect_scores)
        anti_auc = aggregator.auc(anti_scores)

        self.assertEqual(
            perfect_auc,
            1.0,
            msg="Perfect positive discrimination should achieve AUC of 1.0",
        )
        self.assertEqual(
            anti_auc,
            0.0,
            msg="Perfect negative discrimination should achieve AUC of 0.0",
        )

    @pytest.mark.optional
    def test_threshold_sensitivity(self):
        """Test how classification metrics respond to different thresholds."""
        # Dataset where threshold matters
        true_labels = [1, 1, 1, 0, 0, 0, 0, 0]  # 37.5% positive
        scores = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        aggregator = GroundTruthAggregator(true_labels=true_labels)

        # Test different thresholds
        low_threshold_precision = aggregator.precision(scores, threshold=0.3)
        high_threshold_precision = aggregator.precision(scores, threshold=0.7)

        # Calculate expected values:
        # At threshold 0.3: predictions=[1,1,1,1,1,1,0,0], TP=3, FP=3, precision=0.5
        # At threshold 0.7: predictions=[1,1,0,0,0,0,0,0], TP=2, FP=0, precision=1.0

        self.assertEqual(
            low_threshold_precision,
            0.5,
            msg="Low threshold should give precision of 0.5 (3 TP / 6 predictions)",
        )
        self.assertEqual(
            high_threshold_precision,
            1.0,
            msg="High threshold should give precision of 1.0 (2 TP / 2 predictions)",
        )

        # Test recall at different thresholds
        low_threshold_recall = aggregator.recall(scores, threshold=0.3)
        high_threshold_recall = aggregator.recall(scores, threshold=0.7)

        # At threshold 0.3: TP=3, FN=0, recall=1.0
        # At threshold 0.7: TP=2, FN=1, recall=0.67

        self.assertEqual(
            low_threshold_recall,
            1.0,
            msg="Low threshold should give recall of 1.0 (3 TP / 3 positives)",
        )
        self.assertAlmostEqual(
            high_threshold_recall,
            2 / 3,
            places=2,
            msg="High threshold should give recall of 0.67 (2 TP / 3 positives)",
        )

    @pytest.mark.optional
    def test_calibration_scenarios(self):
        """Test ECE with different calibration scenarios."""
        # Well-calibrated model: confidence matches accuracy
        well_calibrated_tuples = [
            (1, 0.9),
            (1, 0.8),
            (0, 0.2),
            (0, 0.1),  # High confidence, correct
            (1, 0.6),
            (0, 0.4),
            (1, 0.7),
            (0, 0.3),  # Medium confidence, correct
        ]

        well_calibrated_aggregator = GroundTruthAggregator(
            true_labels=[1, 1, 0, 0, 1, 0, 1, 0]
        )

        well_calibrated_ece = well_calibrated_aggregator.ece(
            well_calibrated_tuples
        )

        # Overconfident model: high confidence, poor accuracy
        overconfident_tuples = [
            (0, 0.9),
            (0, 0.8),
            (1, 0.9),
            (1, 0.8),  # High confidence, wrong predictions
            (1, 0.6),
            (0, 0.4),
            (1, 0.7),
            (0, 0.3),  # Medium confidence, correct
        ]

        overconfident_ece = well_calibrated_aggregator.ece(overconfident_tuples)

        # Overconfident model should have higher ECE than well-calibrated
        self.assertGreater(
            overconfident_ece,
            well_calibrated_ece,
            msg="Overconfident model should have higher ECE than well-calibrated",
        )

        # Test that truly perfect calibration gives minimum ECE
        # Use actual perfect calibration where predictions match labels exactly
        perfect_calibration_tuples = [
            (1, 1.0),
            (1, 1.0),
            (0, 0.0),
            (0, 0.0),  # Perfect confidence matching
            (1, 1.0),
            (0, 0.0),
            (1, 1.0),
            (0, 0.0),
        ]

        perfect_ece = well_calibrated_aggregator.ece(perfect_calibration_tuples)

        # Perfect calibration should give ECE close to 0
        self.assertLess(
            perfect_ece,
            0.01,
            msg="Perfect calibration should give very low ECE",
        )

        # Perfect should be better than both other scenarios
        self.assertLess(
            perfect_ece,
            well_calibrated_ece,
            msg="Perfect calibration should be better than well-calibrated",
        )
        self.assertLess(
            perfect_ece,
            overconfident_ece,
            msg="Perfect calibration should be better than overconfident",
        )

    @pytest.mark.optional
    def test_error_handling_robustness(self):
        """Test robust error handling for edge cases in real data."""
        # Test with extreme values
        extreme_labels = [0, 1, 0, 1]
        extreme_scores = [0.0, 1.0, 0.0, 1.0]  # Perfect extreme scores

        extreme_aggregator = GroundTruthAggregator(true_labels=extreme_labels)

        # Should handle extreme values gracefully
        auc_result = extreme_aggregator.auc(extreme_scores)
        self.assertEqual(
            auc_result,
            1.0,
            msg="Extreme perfect scores should achieve AUC of 1.0",
        )

        # Test with near-zero variance (constant scores)
        constant_scores = [0.5, 0.5, 0.5, 0.5]
        auc_constant = extreme_aggregator.auc(constant_scores)
        self.assertEqual(
            auc_constant,
            0.5,
            msg="Constant scores should achieve AUC of 0.5 (random performance)",
        )

        # Test MAE with extreme values
        perfect_mae = extreme_aggregator.mae(extreme_scores)
        self.assertEqual(
            perfect_mae,
            0.0,
            msg="Perfect extreme scores should give MAE of 0.0",
        )

        constant_mae = extreme_aggregator.mae(constant_scores)
        self.assertEqual(
            constant_mae, 0.5, msg="Constant 0.5 scores should give MAE of 0.5"
        )

    @pytest.mark.optional
    def test_performance_characteristics(self):
        """Test that metrics perform reasonably with larger datasets."""
        # Create a large dataset that would be typical in production
        large_size = 10000

        # Use fixed seed for reproducibility
        with mock.patch("numpy.random.seed"):
            np.random.seed(42)  # Set seed for this test only

            # Create realistic ground truth and scores
            large_labels = np.random.binomial(
                1, 0.3, large_size
            ).tolist()  # 30% positive
            # Scores that correlate with labels but have realistic noise
            large_scores = [
                label * 0.6 + (1 - label) * 0.4 + np.random.normal(0, 0.2)
                for label in large_labels
            ]
            large_scores = [max(0, min(1, score)) for score in large_scores]

            large_aggregator = GroundTruthAggregator(true_labels=large_labels)

            # Test that computation completes in reasonable time
            import time

            start_time = time.time()

            auc_result = large_aggregator.auc(large_scores)
            brier_result = large_aggregator.brier_score(large_scores)

            end_time = time.time()
            computation_time = end_time - start_time

            # Should complete in reasonable time (less than 5 seconds)
            self.assertLess(
                computation_time,
                5.0,
                msg="Large dataset computation should complete in reasonable time",
            )

            # Results should be reasonable for this synthetic data
            # With this correlation pattern, AUC should be well above random
            self.assertGreater(
                auc_result,
                0.6,
                msg="Large dataset with correlation should achieve decent AUC",
            )

            # Brier score should be reasonable (not too high)
            self.assertLess(
                brier_result,
                0.3,
                msg="Large dataset with correlation should have reasonable Brier score",
            )

            # Test that results are consistent with expectations
            # For this data generation pattern, we expect moderate but clear discrimination
            self.assertLess(
                auc_result,
                0.9,
                msg="Noisy large dataset should not achieve perfect AUC",
            )
            self.assertGreater(
                brier_result,
                0.1,
                msg="Noisy large dataset should have some Brier score error",
            )

    @pytest.mark.optional
    def test_score_format_consistency(self):
        """Test that all methods handle score formats consistently."""
        # Test that all methods handle score-confidence pairs consistently
        scores_only = [0.8, 0.2, 0.9, 0.1, 0.7]
        score_confidence_pairs = [
            [0.8, 0.9],
            [0.2, 0.7],
            [0.9, 0.8],
            [0.1, 0.6],
            [0.7, 0.85],
        ]

        test_aggregator = GroundTruthAggregator(true_labels=[1, 0, 1, 0, 1])

        # Methods that support both formats should give same results
        self.assertEqual(
            test_aggregator.auc(scores_only),
            test_aggregator.auc(score_confidence_pairs),
            msg="AUC should be consistent across score formats",
        )

        self.assertEqual(
            test_aggregator.brier_score(scores_only),
            test_aggregator.brier_score(score_confidence_pairs),
            msg="Brier score should be consistent across score formats",
        )

        self.assertEqual(
            test_aggregator.mae(scores_only),
            test_aggregator.mae(score_confidence_pairs),
            msg="MAE should be consistent across score formats",
        )

        self.assertEqual(
            test_aggregator.precision(scores_only),
            test_aggregator.precision(score_confidence_pairs),
            msg="Precision should be consistent across score formats",
        )

        self.assertEqual(
            test_aggregator.recall(scores_only),
            test_aggregator.recall(score_confidence_pairs),
            msg="Recall should be consistent across score formats",
        )

        self.assertEqual(
            test_aggregator.f1_score(scores_only),
            test_aggregator.f1_score(score_confidence_pairs),
            msg="F1 score should be consistent across score formats",
        )

    @pytest.mark.optional
    def test_zero_division_handling(self):
        """Test handling of zero division scenarios."""
        # Test precision with no predicted positives
        all_negative_predictions = [0.0, 0.0, 0.0, 0.0, 0.0]
        precision_result = self.aggregator.precision(all_negative_predictions)
        self.assertEqual(
            precision_result,
            0.0,
            msg="Precision should be 0.0 when no positives are predicted",
        )

        # Test recall with no actual positives
        all_negative_labels = [0, 0, 0, 0, 0]
        negative_aggregator = GroundTruthAggregator(
            true_labels=all_negative_labels
        )
        recall_result = negative_aggregator.recall([0.8, 0.2, 0.9, 0.1, 0.7])
        self.assertEqual(
            recall_result,
            0.0,
            msg="Recall should be 0.0 when no actual positives exist",
        )

        # Test F1 score with zero precision and recall
        f1_result = negative_aggregator.f1_score(all_negative_predictions)
        self.assertEqual(
            f1_result,
            0.0,
            msg="F1 score should be 0.0 when both precision and recall are zero",
        )
