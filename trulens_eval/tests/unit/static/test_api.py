"""API tests.

These make sure components considered high or low level API are accessible.
"""

import inspect
from pathlib import Path
from typing import Dict
from unittest import main

from tests.test import JSONTestCase, optional_test
from tests.utils import Member
from tests.utils import get_module_members
from tests.utils import get_module_names, get_class_members
from tests.utils import type_str
import trulens_eval


class TestAPI(JSONTestCase):
    """API Tests."""

    def setUp(self):
        pass

    def get_current_members(self) -> Dict[str, Dict[str, Member]]:
        """Get the API members of the current trulens_eval module."""

        objects = {}

        high_classes = set()
        low_classes = set()

        # Enumerate all trulens_eval modules:
        for modname in get_module_names(
            Path(trulens_eval.__file__).parent.parent, matching="trulens_eval"
        ):
            mod = get_module_members(modname)
            if mod is None:
                continue

            highs = {}
            lows = {}

            for mem in mod.api_highs:
                highs[mem.qualname] = type_str(mem.typ)
                if inspect.isclass(mem.val):
                    high_classes.add(mem.val)

            for mem in mod.api_lows:
                lows[mem.qualname] = type_str(mem.typ)
                if inspect.isclass(mem.val):
                    low_classes.add(mem.val)

            objects["module " + modname] = {"highs": highs, "lows": lows}

        # Enumerate all public classes found in the prior step.
        for classes, api_level in zip(
            [high_classes, low_classes], ["high", "low"]
        ):
            for class_ in classes:
                members = get_class_members(class_, class_api_level=api_level)

                highs = {}
                lows = {}

                for mem in members.api_highs:
                    highs[mem.qualname] = type_str(mem.typ)
                for mem in members.api_lows:
                    lows[mem.qualname] = type_str(mem.typ)

                objects["class " + type_str(class_)] = {
                    "highs": highs,
                    "lows": lows,
                }

        return objects

    @optional_test
    def test_apis(self):
        """Check that all high and low level API members are present."""

        members = self.get_current_members()

        self.assertGoldenJSONEqual(
            actual=members,
            golden_filename="api.yaml",
        )


if __name__ == "__main__":
    main()
