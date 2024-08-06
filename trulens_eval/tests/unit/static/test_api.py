"""API tests.

These make sure components considered high or low level API are accessible.
"""

from pathlib import Path
import sys
from unittest import TestCase
from unittest import main

from trulens_eval.tests.test import module_installed
from trulens_eval.tests.test import optional_test
from trulens_eval.tests.test import requiredonly_test
import trulens_eval
from trulens_eval.instruments import Instrument
from trulens_eval.tests.utils import get_module_members
from trulens_eval.utils.imports import Dummy
from trulens_eval.utils.imports import get_module_names


class TestAPI(TestCase):
    """API Tests."""

    def setUp(self):
        pass

    def get_current_members(self):
        """Get the API members of the current trulens_eval module."""

        modules = {}

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
                highs[mem.qualname] = mem

            for mem in mod.api_highs:
                lows[mem.qualname] = mem

            modules[modname] = {"highs": highs, "lows": lows}


if __name__ == "__main__":
    main()
