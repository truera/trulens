"""Validation tests for exported feedback template definitions."""

from collections.abc import Iterable
from string import Formatter
import unittest

from trulens.feedback import templates
from trulens.feedback.templates import base as templates_base

_PROMPT_ATTRIBUTES = ("system_prompt", "user_prompt", "prompt")
_ALLOWED_PROMPT_FIELDS = {
    "context",
    "hypothesis",
    "key_point",
    "max_score",
    "min_score",
    "premise",
    "prompt",
    "question",
    "response",
    "source",
    "statement",
    "statements",
    "submission",
    "summary",
    "text",
    "trace",
}


def _exported_prompt_template_classes() -> (
    Iterable[type[templates_base.FeedbackTemplate]]
):
    """Yield exported template classes that define a system prompt."""
    for symbol in templates.__all__:
        value = getattr(templates, symbol)
        if (
            isinstance(value, type)
            and issubclass(value, templates_base.FeedbackTemplate)
            and value is not templates_base.FeedbackTemplate
            and hasattr(value, "system_prompt")
        ):
            yield value


class TestFeedbackTemplateValidation(unittest.TestCase):
    """Validate prompt template definitions exported by the package."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.template_classes = tuple(_exported_prompt_template_classes())

    def test_finds_expected_template_classes(self) -> None:
        template_names = {
            template.__name__ for template in self.template_classes
        }

        self.assertIn("Groundedness", template_names)
        self.assertIn("ContextRelevance", template_names)
        self.assertIn("Stereotypes", template_names)
        self.assertIn("Helpfulness", template_names)
        self.assertIn("LogicalConsistency", template_names)

    def test_template_classes_are_instantiable_pydantic_models(self) -> None:
        for template in self.template_classes:
            with self.subTest(template=template.__name__):
                instance = template()

                self.assertIsInstance(instance, templates_base.FeedbackTemplate)

    def test_system_prompts_are_non_empty_strings(self) -> None:
        for template in self.template_classes:
            with self.subTest(template=template.__name__):
                system_prompt = template.system_prompt

                self.assertIsInstance(system_prompt, str)
                self.assertGreater(len(system_prompt.strip()), 0)

    def test_user_prompts_are_non_empty_when_declared(self) -> None:
        for template in self.template_classes:
            if not hasattr(template, "user_prompt"):
                continue

            with self.subTest(template=template.__name__):
                user_prompt = template.user_prompt

                self.assertIsInstance(user_prompt, str)
                self.assertGreater(len(user_prompt.strip()), 0)

    def test_criteria_mixins_define_criteria(self) -> None:
        for template in self.template_classes:
            if not issubclass(
                template, templates_base.CriteriaOutputSpaceMixin
            ):
                continue

            with self.subTest(template=template.__name__):
                criteria = template.criteria

                self.assertIsInstance(criteria, str)
                self.assertGreater(len(criteria.strip()), 0)

    def test_output_spaces_are_valid_variants(self) -> None:
        valid_output_spaces = set(templates_base.OutputSpace.__members__)

        for template in self.template_classes:
            if not hasattr(template, "output_space"):
                continue

            with self.subTest(template=template.__name__):
                self.assertIn(template.output_space, valid_output_spaces)

    def test_prompts_only_contain_expected_runtime_fields(self) -> None:
        for template in self.template_classes:
            for prompt_attribute in _PROMPT_ATTRIBUTES:
                prompt = getattr(template, prompt_attribute, None)
                if not isinstance(prompt, str):
                    continue

                with self.subTest(
                    template=template.__name__,
                    prompt_attribute=prompt_attribute,
                ):
                    fields = {
                        field_name
                        for _, field_name, _, _ in Formatter().parse(prompt)
                        if field_name is not None
                    }

                    self.assertLessEqual(fields, _ALLOWED_PROMPT_FIELDS)


if __name__ == "__main__":
    unittest.main()
