"""Regression guard for the methods_to_instrument path-existence check.

TruApp.__init__ checks whether an instrumented method's component exists in the
serialized app before adding a placeholder. It used `next(full_path(json))`,
but a Lens is not callable as a lookup: `Lens.__call__` only accepts a path
ending in `collect` and otherwise raises TypeError. So the check always failed,
the "Added method ... under component" success branch was dead, and users saw a
spurious "App has no component at path" warning even when the component existed.
The fix uses `full_path.get(json)`, matching the sibling check in the same
method. This test pins the Lens contract that makes `.get` mandatory.
"""

from __future__ import annotations

import unittest

from trulens.core.utils.serial import Lens


class TestLensLookupContract(unittest.TestCase):
    def test_get_reads_existing_path_but_call_raises(self):
        json = {"app": {"component": {"leaf": 42}}}
        full_path = Lens().app.component.leaf

        # The read the fix relies on.
        self.assertEqual(next(full_path.get(json)), 42)

        # Calling a non-`collect` Lens is not a lookup and raises, which is why
        # the existence check must use `.get`, not `__call__`.
        with self.assertRaises(TypeError):
            next(full_path(json))


if __name__ == "__main__":
    unittest.main()
