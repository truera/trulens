"""
Static tests, i.e. ones that don't run anything substatial. This should find
issues that occur from merely importing trulens.
"""

from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase

from tests.unit.test import optional_test, requiredonly_test


pp = PrettyPrinter()

# Importing any of these should be ok regardless of optional packages.
base_mods = [
    "trulens_eval",
    "trulens_eval.feedback",
    "trulens_eval.feedback.provider",
    "trulens_eval.feedback.provider.endpoint"
]

# Importing any of these should throw ImportError if optional packages are not
# installed.
optional_mods = [
    "trulens_eval.tru_llama",
    "trulens_eval.feedback.bedrock",
    "trulens_eval.feedback.provider.bedrock",
    "trulens_eval.feedback.provider.endpoint.bedrock",
    "trulens_eval.feedback.provider.litellm",
    "trulens_eval.feedback.provider.endpoint.litellm",
    "trulens_eval.feedback.provider.openai",
    "trulens_eval.feedback.provider.endpoint.openai"
    ]

class TestStatic(TestCase):

    def setUp(self):
        pass

    def test_import_base(self):
        """
        Check that all of the base modules that do not depend on optional
        packages can be imported.
        """

        for mod in base_mods:
            with self.subTest(mod=mod):
                __import__(mod)
    

    @requiredonly_test
    def test_import_optional_fail(self):
        """
        Check that directly importing a module that depends on an optional
        package throws an import error. This test should happen only if optional
        packages have not been installed.
        """

        for mod in optional_mods:
            with self.subTest(mod=mod):
                with self.assertRaises(ModuleNotFoundError) as context:
                    __import__(mod)

                # Make sure the message is the one we produce as part of the
                # optional imports scheme (see
                # utils/imports.py:format_import_errors).
                assert "You should be able to install" in context.exception.args[0] # message?

    @optional_test
    def test_import_optional_success(self):
        """
        Do the same imports as the prior tests except now expecting success as
        we run this test after installing optional packages.
        """

        for mod in optional_mods:
            with self.subTest(mod=mod):
                __import__(mod)


if __name__ == '__main__':
    main()
