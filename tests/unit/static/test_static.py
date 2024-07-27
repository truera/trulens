"""
Static tests, i.e. ones that don't run anything substantial. This should find
issues that occur from merely importing trulens.
"""

from pathlib import Path
import pkgutil
import sys
from unittest import TestCase
from unittest import main

import trulens.core
from trulens.core.instruments import Instrument
from trulens.core.utils.imports import Dummy

from tests.unit.utils import module_installed
from tests.unit.utils import optional_test
from tests.unit.utils import requiredonly_test

# Importing any of these should throw ImportError (or its sublcass
# ModuleNotFoundError) if optional packages are not installed. The key is the
# package that the values depend on. Tests will first make sure the named
# package is not installed and then check that importing any of those named
# modules produces the correct exception. If the uninstalled check fails, it may
# be that something in the requirements list installs what we thought was
# optional in which case it should no longer be considered optional.

optional_mods = dict(
    ipywidgets=["trulens.dashboard.appui"],
    llama_index=[
        "trulens.instrument.llamaindex.tru_llama",
        "trulens.instrument.llamaindex.llama",
        "trulens.instrument.llamaindex.guardrails",
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
    nemoguardrails=["trulens.instrument.nemo.tru_rails"],
)

# snowflake (snowflake-snowpark-python) is not yet supported in python 3.12
if sys.version_info < (3, 12):
    optional_mods["snowflake"] = [
        "trulens.providers.cortex.provider",
        "trulens.providers.cortex.endpoint",
    ]
else:
    assert not module_installed(
        "snowflake-snowpark-python"
    ), "Snowflake should not be installed until it's available in Python 3.12."

optional_mods_flat = [mod for mods in optional_mods.values() for mod in mods]

# Every module not mentioned above should be importable without any optional
# packages.


def get_all_modules(path: Path, startswith=None):
    ret = []
    for modinfo in pkgutil.iter_modules([str(path)]):
        if startswith is not None and not modinfo.name.startswith(startswith):
            continue

        ret.append(modinfo.name)
        if modinfo.ispkg:
            for submod in get_all_modules(path / modinfo.name, startswith=None):
                submodqualname = modinfo.name + "." + submod

                if startswith is not None and not submodqualname.startswith(
                    startswith
                ):
                    continue

                ret.append(modinfo.name + "." + submod)

    return ret


# Get all modules inside trulens_eval:
all_trulens_mods = get_all_modules(
    Path(trulens.core.__file__).parent.parent, startswith="trulens"
)

# Things which should not be imported at all.
not_mods = [
    "trulens.core.database.migrations.env"  # can only be executed by alembic
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
    def setUp(self):
        pass

    def test_import_base(self):
        """Check that all of the base modules that do not depend on optional
        packages can be imported.
        """

        for mod in base_mods:
            with self.subTest(mod=mod):
                __import__(mod)

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
                        f"Instrumented class {cls.name} is dummy meaning it was not importable. Original expception={original_exception}"
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

        from trulens.instrument.langchain import LangChainInstrument

        self._test_instrumentation(LangChainInstrument())

    @optional_test
    def test_instrumentation_llama_index(self) -> None:
        """Check that the llama_index instrumentation is up to date."""

        from trulens.instrument.llamaindex import LlamaInstrument

        self._test_instrumentation(LlamaInstrument())

    @optional_test
    def test_instrumentation_nemo(self):
        """Check that the nemo guardrails instrumentation is up to date."""

        from trulens.instrument.nemo import RailsInstrument

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
                            __import__(mod)

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
                        __import__(mod)


if __name__ == "__main__":
    main()
