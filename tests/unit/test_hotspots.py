import re
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from trulens.hotspots import HotspotsConfig
from trulens.hotspots import hotspots
from trulens.hotspots.hotspots import FeatureStats
from trulens.hotspots.hotspots import TokenFeature
from trulens.hotspots.hotspots import delta
from trulens.hotspots.hotspots import get_feature_stat
from trulens.hotspots.hotspots import opportunity
from trulens.hotspots.hotspots import utest_z


def clean_up_feature(feat: str) -> str:
    """Helper function to clean up a feature before comparison"""
    # consider only two digits for numerical features
    return re.sub(r"(\.\d{2})\d+$", r"\1", feat)


class TestHotspots(TestCase):
    """Tests for hotspots."""

    def setUp(self):
        """Set up common test fixtures."""
        self.sample_scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        self.sample_ranks = pd.Series([1, 2, 3, 4, 5])
        self.sample_feature = TokenFeature(name="test", value="word")
        # Use values that avoid floating point precision issues
        # num_total=8, overall_average=0.5, num_occurrences=4, average_score=0.25
        # neg_average_score = (8 * 0.5 - 4 * 0.25) / (8 - 4) = (4 - 1) / 4 = 0.75
        # delta = 0.25 - 0.75 = -0.5, opportunity = 0.75 - 0.5 = 0.25
        self.sample_stats = FeatureStats(
            p_value=0.01, average_score=0.25, num_occurrences=4
        )

    def _create_test_dataframe(self, score_values, text_values=None):
        """Helper to create test dataframes."""
        if text_values is None:
            text_values = ["hello", "world", "test", "data", "example"][
                : len(score_values)
            ]
        return pd.DataFrame({"score": score_values, "text": text_values})

    @pytest.mark.optional
    def test_simple(self) -> None:
        """Run the simple tests."""

        from pandas import read_csv
        from trulens.hotspots import HotspotsConfig
        from trulens.hotspots import hotspots

        df = read_csv("tests/files/sample.csv.gz")

        config = HotspotsConfig(score_column="score")

        expected_features = [
            "gold>=1872.81",
            "predicted<1854.47",
            "text:carried",
            "text:know",
            "text:lying",
            "text:taxes",
            "text:ef",
            "text:nn",
            "text:weeks",
            "text:comfort",
            "text:request",
            "text:impossible",
            "text:removed",
            "text:arc",
            "text:highly",
            "text:Yet",
            "text:Congress",
            "text:found",
            "text:established",
            "text:Every",
        ]

        _, _, out = hotspots(config, df)
        self.assertEqual(
            [clean_up_feature(str(o[0])) for o in out], expected_features
        )

        # repeated to make sure df is not changed
        _, _, out = hotspots(config, df)
        self.assertEqual(
            [clean_up_feature(str(o[0])) for o in out], expected_features
        )

        non_standard_num_of_rounds = 8
        less_features_config = HotspotsConfig(
            score_column="score", num_rounds=non_standard_num_of_rounds
        )
        _, _, out = hotspots(less_features_config, df)
        self.assertEqual(
            [clean_up_feature(str(o[0])) for o in out],
            expected_features[:non_standard_num_of_rounds],
        )

        # lower_is_better (which is actually for the right
        # setup for the sample CSV, as it is MAE)
        inverted_expected_features = [
            "gold<1872.81",
            "predicted>=1854.47",
            "text:ma",
            "text:our",
            "text:certainly",
            "text:best",
            "text:lay",
            "text:custom",
            "text:individual",
            "text:sell",
            "text:paper",
            "text:having",
            "text:true",
            "text:7",
            "text:meet",
            "text:difficult",
            "text:Government",
            "text:these",
            "text:ui",
            "text:table",
        ]
        inverted_config = HotspotsConfig(
            score_column="score", higher_is_better=False
        )
        _, _, out = hotspots(inverted_config, df)
        self.assertEqual(
            [clean_up_feature(str(o[0])) for o in out],
            inverted_expected_features,
        )

    @pytest.mark.optional
    def test_get_feature_stat(self):
        """Test get_feature_stat with various inputs."""
        # Test with empty occurrences
        result = get_feature_stat(
            self.sample_ranks, self.sample_scores, self.sample_feature, []
        )
        self.assertEqual(result.num_occurrences, 0)
        self.assertEqual(result.average_score, 0.0)
        # p_value calculation involves statistical functions, so keep assertAlmostEqual
        self.assertAlmostEqual(result.p_value, 0.5, places=1)

        # Test with normal occurrences - indices [0, 2, 4] -> values [0.1, 0.3, 0.5]
        # Average: (0.1 + 0.3 + 0.5) / 3 = 0.9 / 3 = 0.3 (exact)
        occurrences = [0, 2, 4]
        result = get_feature_stat(
            self.sample_ranks,
            self.sample_scores,
            self.sample_feature,
            occurrences,
        )
        self.assertEqual(result.num_occurrences, 3)
        self.assertEqual(result.average_score, 0.3)  # Exact calculation
        self.assertIsInstance(result.p_value, float)

    @pytest.mark.optional
    def test_utest_z(self):
        """Test utest_z function with various edge cases."""
        # Test division by zero cases
        self.assertEqual(utest_z(0.0, 0, 10), 0.0)  # poss = 0
        self.assertEqual(utest_z(15.0, 5, 0), 0.0)  # negs = 0
        self.assertEqual(utest_z(0.0, 0, 0), 0.0)  # both = 0

        # Test normal operation
        result = utest_z(15.0, 5, 5)
        self.assertIsInstance(result, float)

    @pytest.mark.optional
    def test_delta_and_opportunity(self):
        """Test delta and opportunity functions with edge cases."""
        # Test when feature appears in all samples (edge case)
        all_samples_stats = FeatureStats(
            p_value=0.01, average_score=0.7, num_occurrences=10
        )
        self.assertEqual(delta(10, 0.5, all_samples_stats), 0.0)
        self.assertEqual(opportunity(10, 0.5, all_samples_stats), 0.0)

        # Test normal operation with exact calculations
        # num_total=8, overall_average=0.5, num_occurrences=4, average_score=0.25
        # neg_average_score = (8 * 0.5 - 4 * 0.25) / (8 - 4) = (4 - 1) / 4 = 0.75
        # delta = 0.25 - 0.75 = -0.5, opportunity = 0.75 - 0.5 = 0.25
        delta_result = delta(8, 0.5, self.sample_stats)
        opportunity_result = opportunity(8, 0.5, self.sample_stats)

        self.assertEqual(delta_result, -0.5)  # Exact calculation
        self.assertEqual(opportunity_result, 0.25)  # Exact calculation

    @pytest.mark.optional
    def test_hotspots_edge_cases(self):
        """Test hotspots function with edge cases."""
        # Single row dataframe
        single_row_df = self._create_test_dataframe([0.5], ["hello world"])
        config = HotspotsConfig(score_column="score", min_occurrences=1)

        modified_df, avg_score, features = hotspots(config, single_row_df)
        self.assertEqual(len(modified_df), 1)
        self.assertEqual(avg_score, 0.5)
        self.assertIsInstance(features, list)

        # Empty dataframe
        empty_df = self._create_test_dataframe([], [])
        modified_df, avg_score, features = hotspots(config, empty_df)
        self.assertEqual(len(modified_df), 0)
        self.assertTrue(pd.isna(avg_score) or avg_score == 0.0)
        self.assertEqual(len(features), 0)

    @pytest.mark.optional
    def test_hotspots_with_nan_scores(self):
        """Test hotspots handles NaN scores correctly."""
        df = self._create_test_dataframe([0.1, np.nan, 0.3, np.nan, 0.5])
        config = HotspotsConfig(score_column="score")

        modified_df, avg_score, _ = hotspots(config, df)

        # Should filter out NaN scores
        self.assertEqual(len(modified_df), 3)
        # Average: (0.1 + 0.3 + 0.5) / 3 = 0.9 / 3 = 0.3 (exact)
        self.assertEqual(avg_score, 0.3)  # Exact calculation
