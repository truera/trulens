"""API tests.

These make sure components considered high or low level API are accessible.
"""

import inspect
import sys
from typing import Dict, Optional
from unittest import main
from unittest import skip
from unittest import skipIf

from jsondiff import SymmetricJsonDiffSyntax
from jsondiff import diff
from jsondiff.symbols import Symbol
from trulens.core.utils.imports import is_dummy
from trulens.core.utils.serial import Lens

from tests.test import JSONTestCase
from tests.test import optional_test
from tests.utils import Member
from tests.utils import get_class_members
from tests.utils import get_module_members
from tests.utils import get_submodule_names
from tests.utils import type_str


class TestAPI(JSONTestCase):
    """API Tests."""

    def setUp(self):
        self.pyversion = ".".join(map(str, sys.version_info[0:2]))

    def get_members(self, mod) -> Dict[str, Dict[str, Member]]:
        """Get the API members of the trulens_eval module."""
        # TODEP: Deprecate after trulens_eval is removed.

        objects = {}

        classes = set()

        # Enumerate mod and all submodules.
        for modname in get_submodule_names(mod):
            mod = get_module_members(modname)
            if mod is None:
                continue

            highs = {}
            lows = {}

            for mem in mod.api_highs:
                if inspect.isclass(mem.val):
                    classes.add(mem.val)

                highs[mem.name] = type_str(mem.typ)

            for mem in mod.api_lows:
                if inspect.isclass(mem.val):
                    classes.add(mem.val)

                lows[mem.name] = type_str(mem.typ)

            k = modname  # + "(" + type_str(type(mod.obj)) + ")"

            objects[k] = {
                "highs": highs,
                "lows": lows,
                "__class__": type_str(type(mod.obj)),
            }
            if mod.version is not None:
                objects[k]["__version__"] = mod.version

        # Enumerate all public classes found in the prior step.
        for class_ in classes:
            if is_dummy(class_):
                with self.subTest(class_=class_.__name__):
                    self.fail(
                        f"Dummy class found in classes: {str(class_)}. Make sure all optional modules are installed before running this test."
                    )
                # Record this as a test issue but continue to the next class.
                continue

            members = get_class_members(
                class_, class_api_level="low"
            )  # api level is arbitrary

            attrs = {}

            for mem in members.api_lows:  # because of "low" above
                attrs[mem.name] = type_str(mem.typ)

            k = type_str(class_)  # + "(" + type_str(type(class_)) + ")"

            info = {
                "__class__": type_str(type(members.obj)),
                "__bases__": [type_str(base) for base in members.obj.__bases__],
                "attributes": attrs,
            }

            # if k in objects:
            #    self.assertJSONEqual(info, objects[k], path=Lens()[k])
            print(f"duplicate {k}")

            objects[k] = info

        return objects

    def get_members_trulens_eval(self) -> Dict[str, Dict[str, Member]]:
        """Get the API members of the trulens_eval module."""
        # TODEP: Deprecate after trulens_eval is removed.

        import trulens_eval

        return self.get_members(trulens_eval)

    def get_members_trulens(self) -> Dict[str, Dict[str, Member]]:
        """Get the API members of the trulens_eval module."""

        import trulens

        return self.get_members(trulens)

    def _flatten_api_diff(self, diff, lens: Optional[Lens] = None):
        """Flatten the API diff for easier comparison."""

        if lens is None:
            lens = Lens()

        flat_diffs = []

        if isinstance(diff, dict):
            for k, v in diff.items():
                if isinstance(k, Symbol):
                    flat_diffs.append((k, lens, v))
                    continue

                for f in self._flatten_api_diff(v, lens[k]):
                    flat_diffs.append(f)

        return flat_diffs

    @skip("Compat not ready.")
    @skipIf(sys.version_info[0:2] != (3, 11), "Only run on Python 3.11")
    @optional_test
    def test_api_trulens_eval_compat(self):
        """Check that the trulens_eval API members are still present.

        To regenerate golden file, run `make test-write-api`.
        """
        # TODEP: Deprecate after trulens_eval is removed.

        golden_file = f"api.trulens_eval.{self.pyversion}.yaml"

        members = self.get_members_trulens_eval()

        self.write_golden(
            path=golden_file, data=members
        )  # will raise exception if golden file is written

        expected = self.load_golden(golden_file)

        jdiff = diff(expected, members, syntax=SymmetricJsonDiffSyntax())
        flat_diffs = self._flatten_api_diff(jdiff)

        if flat_diffs:
            for diff_type, diff_lens, diff_value in flat_diffs:
                with self.subTest(api=str(diff_lens)):
                    self.fail(
                        f"trulens_eval compatibility API mismatch: {diff_type} at {diff_lens} value {diff_value}"
                    )

    @skipIf(sys.version_info[0:2] != (3, 11), "Only run on Python 3.11")
    @optional_test
    def test_api_trulens(self):
        """Check that the trulens API members are still present.

        To regenerate golden file, run `make test-write-api`.
        """

        golden_file = f"api.trulens.{self.pyversion}.yaml"

        members = self.get_members_trulens()

        self.write_golden(
            path=golden_file, data=members
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
