"""Unit tests for ``trulens.dashboard.utils.dashboard_utils``."""

import pandas as pd
from trulens.dashboard.utils.dashboard_utils import get_unique_app_tags


class TestGetUniqueAppTags:
    """Tests for ``get_unique_app_tags`` (regression for issue #1689).

    Each app version's ``tags`` value is a single string, so the tag filter
    must collect whole tag strings. The previous ``set(app_version["tags"])``
    iterated a string into its individual characters.
    """

    def test_tags_are_not_split_into_characters(self):
        # Regression for #1689: a tag string used to be exploded into single
        # characters (e.g. "production" -> "p", "r", "o", "d", ...).
        df = pd.DataFrame({"tags": ["production", "staging"]})
        assert get_unique_app_tags(df) == ["production", "staging"]

    def test_single_tag_kept_whole(self):
        # Even one app with one tag must stay intact, not become characters.
        df = pd.DataFrame({"tags": ["production"]})
        assert get_unique_app_tags(df) == ["production"]

    def test_duplicate_tags_deduplicated_and_sorted(self):
        df = pd.DataFrame({"tags": ["prod", "dev", "prod", "abc"]})
        assert get_unique_app_tags(df) == ["abc", "dev", "prod"]

    def test_empty_tag_strings_skipped(self):
        # "" is the default tag value and must not appear as a blank option.
        df = pd.DataFrame({"tags": ["prod", "", "dev", ""]})
        assert get_unique_app_tags(df) == ["dev", "prod"]

    def test_all_empty_yields_no_tags(self):
        df = pd.DataFrame({"tags": ["", ""]})
        assert get_unique_app_tags(df) == []
