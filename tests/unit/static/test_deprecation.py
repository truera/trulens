"""
Derpecation tests.
"""

from enum import Enum
import importlib
import inspect
import sys
from unittest import TestCase
from unittest import main

from tests.test import optional_test


class TestDeprecation(TestCase):
    """Tests for deprecation of old module names."""

    def setUp(self):
        # Pre-trulens package aliases. These are modules that have an
        # __init__.py that need to issue warnings when imported and have the old
        # contents that also issue warnings when used.

        self.trulens_eval_modules = {
            "trulens_eval": [
                "Tru",  # main interface
                # app types
                "TruBasicApp",
                "TruCustomApp",
                "TruChain",
                "TruLlama",
                "TruVirtual",
                "TruRails",
                # app setup
                "FeedbackMode",
                # feedback setup
                "Feedback",
                "Select",
                # feedback providers
                "Provider",
                "AzureOpenAI",
                "OpenAI",
                "Langchain",
                "LiteLLM",
                "Bedrock",
                "Huggingface",
                "HuggingfaceLocal",
                "Cortex",
                # misc utility
                "TP",
            ],
            "trulens_eval.database.migrations": [],  # no public names
            "trulens_eval.feedback.provider": [
                "Provider",
                "OpenAI",
                "AzureOpenAI",
                "Huggingface",
                "HuggingfaceLocal",
                "LiteLLM",
                "Bedrock",
                "Langchain",
                "Cortex",
            ],
            "trulens_eval.feedback.provider.endpoint": [
                "Endpoint",
                "DummyEndpoint",
                "HuggingfaceEndpoint",
                "OpenAIEndpoint",
                "LiteLLMEndpoint",
                "BedrockEndpoint",
                "OpenAIClient",
                "LangchainEndpoint",
                "CortexEndpoint",
            ],
            "trulens_eval.feedback": [
                "Feedback",
                "Embeddings",
                "GroundTruthAgreement",
                "OpenAI",
                "AzureOpenAI",
                "Huggingface",
                "HuggingfaceLocal",
                "LiteLLM",
                "Bedrock",
                "Langchain",
                "Cortex",
            ],
            "trulens_eval.schema": [],  # no names
            "trulens_eval.react_components.record_viewer": ["record_viewer"],
        }

    @optional_test
    def test_init_aliases(self):
        """Check that all trulens_eval.*.__init__ aliases are still usable
        produce deprecation messages when used.

        Also checks that importing the module itself produces a deprecation
        warning.
        """

        for modname, names in self.trulens_eval_modules.items():
            with self.subTest(modname=modname):
                # Make sure importing the module shows a warning:
                with self.subTest("module deprecation warning"):
                    with self.assertWarns(DeprecationWarning):
                        mod = None
                        if modname in sys.modules:
                            # Delete the module in case already imported. This
                            # is to catch the deprecation warning which only
                            # occurs the first time it is imported.
                            del sys.modules[modname]

                        mod = importlib.import_module(
                            modname,
                        )

                if mod is None:
                    continue

                for name in names:
                    if name in [
                        "Cortex",
                        "CortexEndpoint",
                        "TruRails",
                    ] and sys.version_info >= (3, 12):
                        # These require python 3.12 .
                        continue

                    with self.subTest(name=name):
                        # Can get the named object from module:
                        self.assertTrue(
                            hasattr(mod, name),
                            f"Module {modname} does not have alias {name}.",
                        )

                        val = getattr(mod, name)

                        if inspect.isclass(val) and issubclass(val, Enum):
                            # The deprecation warning scheme does not work for Enums.
                            continue

                        with self.subTest("alias call deprecation warning"):
                            with self.assertWarns(DeprecationWarning):
                                # try calling it:
                                try:
                                    val()  # will most likely fail, but should do so after the deprecation message
                                except BaseException:
                                    pass


if __name__ == "__main__":
    main()
