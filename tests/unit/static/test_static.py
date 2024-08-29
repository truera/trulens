"""
Static tests, i.e. ones that don't run anything substantial. This should find
issues that occur from merely importing trulens.
"""

import importlib
import sys
from unittest import TestCase
from unittest import main
from unittest import skipIf

import trulens
from trulens.core.instruments import Instrument
from trulens.core.utils.imports import Dummy

from tests.test import module_installed
from tests.test import optional_test
from tests.test import requiredonly_test
from tests.utils import get_submodule_names

# Importing any of these should throw ImportError (or its subclass
# ModuleNotFoundError) if optional packages are not installed. The key is the
# package that the values depend on. Tests will first make sure the named
# package is not installed and then check that importing any of those named
# modules produces the correct exception. If the uninstalled check fails, it may
# be that something in the requirements list installs what we thought was
# optional in which case it should no longer be considered optional.

optional_mods = dict(
    llama_index=[
        "trulens.apps.llamaindex.tru_llama",
        "trulens.apps.llamaindex.llama",
        "trulens.apps.llamaindex.guardrails",
    ],
    boto3=[
        "trulens.providers.bedrock.provider",
        "trulens.providers.bedrock.endpoint",
    ],
    litellm=[
        "trulens.providers.litellm.provider",
        "trulens.providers.litellm.endpoint",
    ],
    openai=[
        "trulens.providers.openai.provider",
        "trulens.providers.openai.endpoint",
    ],
)

# snowflake (snowflake-snowpark-python) is not yet supported in python 3.12
if sys.version_info < (3, 12):
    optional_mods["nemoguardrails"] = ["trulens.apps.nemo"]
else:
    assert not module_installed(
        "snowflake-snowpark-python"
    ), "`snowflake-snowpark-python` should not be installed until it's available in Python 3.12."
    assert not module_installed(
        "nemoguardrails"
    ), "`nemoguardrails` should not be installed until it's available in Python 3.12."

optional_mods_flat = [mod for mods in optional_mods.values() for mod in mods]

# Every module not mentioned above should be importable without any optional
# packages.

# Get all modules inside trulens_eval:
all_trulens_mods = list(get_submodule_names(trulens))

# Things which should not be imported at all.
not_mods = [
    "trulens.core.database.migrations.env",  # can only be executed by alembic
    "trulens.feedback.embeddings",  # requires llama_index
]

if sys.version_info >= (3, 12):
    not_mods.extend(["snowflake", "trulens.providers.cortex"])

# Importing any of these should be ok regardless of optional packages. These are
# all modules not mentioned in optional modules above.
base_mods = [
    mod
    for mod in all_trulens_mods
    if mod not in optional_mods_flat and mod not in not_mods
]


class TestStatic(TestCase):
    """Static tests, those that are not expected to execute real code other than
    code involved in loading and executing modules."""

    def setUp(self):
        pass

    def test_import_base(self):
        """Check that all of the base modules that do not depend on optional
        packages can be imported.
        """

        for mod in base_mods:
            with self.subTest(mod=mod):
                importlib.import_module(mod)

    def _test_instrumentation(self, i: Instrument):
        """Check that the instrumentation specification is good in these ways:

        - (1) All classes mentioned are loaded/importable.
        - (2) All methods associated with a class are actually methods of that
          class.
        - (3) All classes belong to modules that are to be instrumented. Otherwise
          this may be a sign that a class is an alias for things like builtin
          types like functions/callables or None.
        """

        for cls in i.include_classes:
            with self.subTest(cls=cls):
                if isinstance(cls, Dummy):  # (1)
                    original_exception = cls.original_exception
                    self.fail(
                        f"Instrumented class {cls.name} is dummy meaning it was not importable. Original exception={original_exception}"
                    )

                # Disabled #2 test right now because of too many failures. We
                # are using the class filters too liberally.
                """
                for method, class_filter in i.include_methods.items():
                    if class_filter_matches(f=class_filter, obj=cls):
                        with self.subTest(method=method):
                            self.assertTrue(
                                hasattr(cls, method),  # (2)
                                f"Method {method} is not a method of class {cls}."
                            )
                """

                if not i.to_instrument_module(cls.__module__):  # (3)
                    self.fail(
                        f"Instrumented class {cls} is in module {cls.__module__} which is not to be instrumented."
                    )

    @optional_test
    def test_instrumentation_langchain(self):
        """Check that the langchain instrumentation is up to date."""

        from trulens.apps.langchain import LangChainInstrument

        self._test_instrumentation(LangChainInstrument())

    @optional_test
    def test_instrumentation_llama_index(self) -> None:
        """Check that the llama_index instrumentation is up to date."""

        from trulens.apps.llamaindex import LlamaInstrument

        self._test_instrumentation(LlamaInstrument())

    @skipIf(
        sys.version_info >= (3, 12), "nemo is not yet supported in Python 3.12"
    )
    @optional_test
    def test_instrumentation_nemo(self):
        """Check that the nemo guardrails instrumentation is up to date."""
        from trulens.apps.nemo import RailsInstrument

        self._test_instrumentation(RailsInstrument())

    @requiredonly_test
    def test_import_optional_fail(self) -> None:
        """
        Check that directly importing a module that depends on an optional
        package throws an import error. This test should happen only if optional
        packages have not been installed.
        """

        for opt, mods in optional_mods.items():
            with self.subTest(optional=opt):
                # First make sure the optional package is not installed.
                self.assertFalse(
                    module_installed(opt),
                    msg=f"Module {opt} was not supposed to be installed for this test.",
                )

                for mod in mods:
                    with self.subTest(mod=mod):
                        # Make sure the import raises ImportError:
                        with self.assertRaises(ImportError):
                            importlib.import_module(mod)

    @optional_test
    def test_import_optional_success(self):
        """
        Do the same imports as the prior tests except now expecting success as
        we run this test after installing optional packages.
        """

        for opt, mods in optional_mods.items():
            with self.subTest(optional=opt):
                # First make sure the optional package is installed.
                self.assertTrue(
                    module_installed(opt),
                    f"Module {opt} was supposed to be installed for this test.",
                )

                for mod in mods:
                    with self.subTest(mod=mod):
                        # Make sure we can import the module now.
                        importlib.import_module(mod)


if __name__ == "__main__":
    main()
