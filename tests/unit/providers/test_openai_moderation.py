"""Regression test: each moderation_* method must report its own category.

moderation_harassment_threatening read category_scores.harassment (a copy-paste
of moderation_harassment) instead of harassment_threatening, so it silently
returned the plain harassment score.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    from trulens.providers.openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class TestOpenAIModerationCategories(unittest.TestCase):
    def setUp(self):
        if OpenAI is None:
            self.skipTest("trulens-providers-openai not available.")

    def _provider(self, **category_scores):
        provider = OpenAI.__new__(OpenAI)  # skip __init__ (needs an API key)
        response = SimpleNamespace(
            category_scores=SimpleNamespace(**category_scores)
        )
        provider._moderation = lambda text: response
        return provider

    def test_harassment_and_threatening_read_distinct_fields(self):
        provider = self._provider(harassment=0.11, harassment_threatening=0.99)
        self.assertEqual(provider.moderation_harassment("x"), 0.11)
        self.assertEqual(provider.moderation_harassment_threatening("x"), 0.99)


if __name__ == "__main__":
    unittest.main()
