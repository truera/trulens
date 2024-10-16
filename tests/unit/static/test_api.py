"""API tests.

These make sure components considered high or low level API are accessible.
"""

import inspect
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from unittest import main
from unittest import skipIf

from jsondiff import SymmetricJsonDiffSyntax
from jsondiff import diff
from jsondiff.symbols import Symbol
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import imports as import_utils
from trulens.core.utils import serial as serial_utils

from tests import test as mod_test
from tests import utils as test_utils

_GOLDEN_DIRECTORY = "tests/unit/static/golden"


class TestAPI(mod_test.TruTestCase):
    """API Tests."""

    def setUp(self):
        self.pyversion = ".".join(map(str, sys.version_info[0:2]))

    def get_members(
        self, mod, aliases_are_defs: bool = False
    ) -> Dict[str, Dict[str, test_utils.Member]]:
        """Get the API members of the trulens_eval module."""
        # TODEP: Deprecate after trulens_eval is removed.

        objects = {}

        classes = set()

        # Enumerate mod and all submodules.
        for modname in test_utils.get_submodule_names(mod):
            mod = test_utils.get_module_members(
                modname, aliases_are_defs=aliases_are_defs
            )
            if mod is None:
                continue

            highs = {}
            lows = {}

            for mem in mod.api_highs:
                if inspect.isclass(mem.val):
                    classes.add((modname + "." + mem.name, mem.val))

                highs[mem.name] = test_utils.type_str(mem.typ)

            for mem in mod.api_lows:
                if inspect.isclass(mem.val):
                    classes.add((modname + "." + mem.name, mem.val))

                lows[mem.name] = test_utils.type_str(mem.typ)

            k = modname

            objects[k] = {
                "highs": highs,
                "lows": lows,
                "__class__": test_utils.type_str(type(mod.obj)),
            }
            if mod.version is not None:
                objects[k]["__version__"] = mod.version

        # Enumerate all public classes found in the prior step.
        for class_alias, class_ in classes:
            if import_utils.is_dummy(class_):
                if not deprecation_utils.is_deprecated(class_):
                    with self.subTest(class_=class_.__name__):
                        self.fail(
                            f"Dummy class found in classes: {str(class_)}. Make sure all optional modules are installed before running this test."
                        )
                # Was dummy, but a deprecation dummy. Skip it.
                continue

            members = test_utils.get_class_members(
                class_,
                class_api_level="low",
                class_alias=class_alias,
                overrides_are_defs=aliases_are_defs,
            )  # api level is arbitrary

            attrs = {}

            for mem in members.api_lows:  # because of "low" above
                attrs[mem.name] = test_utils.type_str(mem.typ)

            if aliases_are_defs:
                k = class_alias
            else:
                k = test_utils.type_str(class_)

            info = {
                "__class__": test_utils.type_str(type(members.obj)),
                "__bases__": [
                    test_utils.type_str(base) for base in members.obj.__bases__
                ],
                "attributes": attrs,
            }

            objects[k] = info

        return objects

    def get_members_trulens_eval(
        self, aliases_are_defs: bool = False
    ) -> Dict[str, Dict[str, test_utils.Member]]:
        """Get the API members of the trulens_eval module."""
        # TODEP: Deprecate after trulens_eval is removed.

        import trulens_eval

        return self.get_members(trulens_eval, aliases_are_defs=aliases_are_defs)

    def get_members_trulens(self) -> Dict[str, Dict[str, test_utils.Member]]:
        """Get the API members of the trulens_eval module."""

        import trulens

        return self.get_members(trulens)

    def _flatten(
        self, val: Any, lens: Optional[serial_utils.Lens] = None
    ) -> Iterable[Tuple[serial_utils.Lens, Any]]:
        """Flatten the API diff for easier comparison."""

        if lens is None:
            lens = serial_utils.Lens()

        if len(lens) > 1 or isinstance(val, (str, int, float, bool)):
            yield (lens, val)
        elif isinstance(val, dict):
            for k, v in val.items():
                yield from self._flatten(val=v, lens=lens[k])
        elif isinstance(val, Sequence):
            for i, v in enumerate(val):
                yield from self._flatten(val=v, lens=lens[i])
        else:
            raise ValueError(f"Unexpected type {type(val)}")

    def _flatten_api_diff(
        self, diff_value: Any, lens: Optional[serial_utils.Lens] = None
    ) -> Iterable[Tuple[Symbol, serial_utils.Lens, Any]]:
        """Flatten the API diff for easier comparison."""

        if lens is None:
            lens = serial_utils.Lens()

        if isinstance(diff_value, dict):
            for k, v in diff_value.items():
                if isinstance(k, Symbol):
                    for sublens, subval in self._flatten(val=v, lens=lens):
                        yield (k, sublens, subval)
                else:
                    yield from self._flatten_api_diff(
                        diff_value=v, lens=lens[k]
                    )

    # @skip("Compat not ready.")
    @skipIf(sys.version_info[0:2] != (3, 11), "Only run on Python 3.11")
    @mod_test.optional_test
    def test_api_trulens_eval_compat(self):
        """Check that the trulens_eval API members are still present.

        To regenerate golden file, run `make write-api`.
        """
        # TODEP: Deprecate after trulens_eval is removed.

        golden_file = (
            f"{_GOLDEN_DIRECTORY}/api.trulens_eval.{self.pyversion}.yaml"
        )

        members = self.get_members_trulens_eval(aliases_are_defs=True)

        self.write_golden(
            golden_path=golden_file, data=members
        )  # will raise exception if golden file is written

        expected = self.load_golden(golden_file)

        jdiff = diff(expected, members, syntax=SymmetricJsonDiffSyntax())

        flat_diffs = list(self._flatten_api_diff(jdiff))

        fh = None
        if os.environ.get("WRITE_API_DIFF"):
            fh = open("api.diff", "w")

        if flat_diffs:
            for diff_type, diff_lens, diff_value in flat_diffs:
                if diff_type.label == "insert":
                    # ignore additions
                    continue
                if repr(diff_lens.path[-1]) == ".__bases__":
                    # Ignore __bases__ differences.
                    continue
                if repr(diff_lens.path[-1]) == ".__class__":
                    # Ignore __class__ differences.
                    continue
                if repr(diff_lens.path[-1]) == ".__version__":
                    # Ignore __version__ differences.
                    continue
                if isinstance(diff_value, dict) and len(diff_value) == 0:
                    # Ignore empty dicts in diffs.
                    continue
                with self.subTest(api=str(diff_lens)):
                    if fh:
                        fh.write(f"{diff_type} {diff_lens} {diff_value}\n")
                    self.fail(
                        f"trulens_eval compatibility API mismatch: {diff_type} at {diff_lens} value {diff_value}"
                    )

        if fh:
            fh.close()

    @skipIf(sys.version_info[0:2] != (3, 11), "Only run on Python 3.11")
    @mod_test.optional_test
    def test_api_trulens(self):
        """Check that the trulens API members are still present.

        To regenerate golden file, run `make write-api`.
        """

        golden_file = f"{_GOLDEN_DIRECTORY}/api.trulens.{self.pyversion}.yaml"

        members = self.get_members_trulens()

        self.write_golden(
            golden_path=golden_file, data=members
        )  # will raise exception if golden file is written

        expected = self.load_golden(golden_file)

        jdiff = diff(expected, members, syntax=SymmetricJsonDiffSyntax())
        flat_diffs = self._flatten_api_diff(jdiff)

        if flat_diffs:
            for diff_type, diff_lens, diff_value in flat_diffs:
                with self.subTest(api=str(diff_lens)):
                    self.fail(
                        f"API mismatch: {diff_type} at {diff_lens} value {diff_value}"
                    )


if __name__ == "__main__":
    main()
