from pathlib import Path
import re
from unittest import TestCase

import pytest


def clean_up_feature(feat: str) -> str:
    """Helper function to clean up a feature before comparison"""
    # consider only two digits for numerical features
    return re.sub(r"(\.\d{2})\d+$", r"\1", feat)


_HOTSPOTS_INIT = (
    Path(__file__).parents[2]
    / "src"
    / "hotspots"
    / "trulens"
    / "hotspots"
    / "__init__.py"
)


class TestHotspots(TestCase):
    """Tests for hotspots."""

    def test_package_deprecation_warning(self) -> None:
        """The package __init__ contains a DeprecationWarning."""
        source = _HOTSPOTS_INIT.read_text()
        self.assertIn("DeprecationWarning", source)
        self.assertIn("The `trulens-hotspots` package is deprecated", source)

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
