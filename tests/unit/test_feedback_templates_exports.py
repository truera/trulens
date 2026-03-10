"""Tests for explicit template-module export surfaces."""

import unittest

from trulens.feedback import templates
from trulens.feedback.templates import agent as templates_agent
from trulens.feedback.templates import quality as templates_quality
from trulens.feedback.templates import rag as templates_rag
from trulens.feedback.templates import safety as templates_safety


class TestFeedbackTemplateExports(unittest.TestCase):
    """Ensure template exports are explicit and intentional."""

    def test_templates_package_re_exports_selected_symbols(self) -> None:
        expected = {
            "Groundedness",
            "ContextRelevance",
            "Stereotypes",
            "Helpfulness",
            "PlanAdherence",
            "FeedbackTemplate",
            "OutputSpace",
        }

        for symbol in expected:
            self.assertIn(symbol, templates.__all__)
            self.assertTrue(hasattr(templates, symbol))

    def test_templates_package_does_not_export_domain_hierarchy(self) -> None:
        # Hierarchy classes are intentionally kept module-internal.
        self.assertNotIn("GroundTruth", templates.__all__)
        self.assertNotIn("Relevance", templates.__all__)
        self.assertNotIn("Moderation", templates.__all__)
        self.assertNotIn("Legality", templates.__all__)
        self.assertNotIn("Hate", templates.__all__)

    def test_domain_modules_define_explicit_all(self) -> None:
        for mod in [
            templates_agent,
            templates_quality,
            templates_rag,
            templates_safety,
        ]:
            self.assertTrue(hasattr(mod, "__all__"))
            self.assertGreater(len(mod.__all__), 0)

    def test_safety_module_all_excludes_internal_hierarchy(self) -> None:
        self.assertNotIn("Moderation", templates_safety.__all__)
        self.assertNotIn("Legality", templates_safety.__all__)
        self.assertNotIn("Hate", templates_safety.__all__)


if __name__ == "__main__":
    unittest.main()
