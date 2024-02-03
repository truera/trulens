"""
Static tests, i.e. ones that don't run anything substatial. This should find
issues that occur from merely importing trulens.
"""

import os
from pathlib import Path
import pkgutil
from pprint import PrettyPrinter
from typing import List
from unittest import main
from unittest import TestCase

from tests.unit.test import module_installed
from tests.unit.test import optional_test
from tests.unit.test import requiredonly_test

import trulens_eval

pp = PrettyPrinter()

# Importing any of these should throw ImportError (or its sublcass
# ModuleNotFoundError) if optional packages are not installed. The key is the
# package that the values depend on. Tests will first make sure the named
# package is not installed and then check that importing any of those named
# modules produces the correct exception. If the uninstalled check fails, it may
# be that something in the requirements list installs what we thought was
# optional in which case it should no longer be considered optional.

optional_mods = dict(
    pinecone=[
        "trulens_eval.Example_TruBot"
    ],
    ipywidgets=[
        "trulens_eval.appui"
    ],
    llama_index = [
        "trulens_eval.tru_llama",
        "trulens_eval.utils.llama"
    ],
    boto3 = [
        "trulens_eval.feedback.provider.bedrock",
        "trulens_eval.feedback.provider.endpoint.bedrock"
    ],
    litellm = [
        "trulens_eval.feedback.provider.litellm",
        "trulens_eval.feedback.provider.endpoint.litellm",
    ],
    openai = [
        "trulens_eval.feedback.provider.openai",
        "trulens_eval.feedback.provider.endpoint.openai"
    ]
)

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

                if startswith is not None and not submodqualname.startswith(startswith):
                    continue
                
                ret.append(modinfo.name + "." + submod)

    return ret

# Get all modules inside trulens_eval: 
all_trulens_mods = get_all_modules(
    Path(trulens_eval.__file__).parent.parent,
    startswith="trulens_eval"
)

# Things which should not be imported at all.
not_mods = [
    "trulens_eval.database.migrations.env" # can only be executed by alembic
]

# Importing any of these should be ok regardless of optional packages. These are
# all modules not mentioned in optional modules above.
base_mods = [
    mod for mod in all_trulens_mods 
    if mod not in optional_mods_flat 
    and mod not in not_mods
]

# OLD list:
"""
    "trulens_eval",
    "trulens_eval.tru",
    "trulens_eval.tru_chain",
    "trulens_eval.tru_basic_app",
    "trulens_eval.tru_custom_app",
    "trulens_eval.tru_virtual",
    "trulens_eval.app",
    "trulens_eval.db",
    "trulens_eval.schema",
    "trulens_eval.keys",
    "trulens_eval.instruments",
    "trulens_eval.feedback",
    "trulens_eval.feedback.provider",
    "trulens_eval.feedback.provider.endpoint"
]
"""

class TestStatic(TestCase):

    def setUp(self):
        pass

    def test_import_base(self):
        """
        Check that all of the base modules that do not depend on optional
        packages can be imported.
        """

        for mod in base_mods:
            with self.subTest(msg=mod):
                __import__(mod)
    

    @requiredonly_test
    def test_import_optional_fail(self):
        """
        Check that directly importing a module that depends on an optional
        package throws an import error. This test should happen only if optional
        packages have not been installed.
        """

        for opt, mods in optional_mods.items():
            with self.subTest(optional=opt):
                # First make sure the optional package is not installed.
                self.assertFalse(module_installed(opt))

                for mod in mods:
                    with self.subTest(mod=mod):
                        # Make sure the import raises ImportError:
                        with self.assertRaises(ImportError) as context:
                            __import__(mod)

                        # Make sure the message in the exception is the one we
                        # produce as part of the optional imports scheme (see
                        # utils/imports.py:format_import_errors).
                        self.assertIn("You should be able to install", context.exception.args[0])

    @optional_test
    def test_import_optional_success(self):
        """
        Do the same imports as the prior tests except now expecting success as
        we run this test after installing optional packages.
        """

        for opt, mods in optional_mods.items():
            with self.subTest(optional=opt):
                # First make sure the optional package is installed.
                self.assertTrue(module_installed(opt))

                for mod in mods:
                    with self.subTest(mod=mod):
                        # Make sure we can import the module now.                        
                        __import__(mod)


if __name__ == '__main__':
    main()
